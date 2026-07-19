package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import br.com.guialves.rflr.datastructure.MinSegmentTree;
import br.com.guialves.rflr.datastructure.SumSegmentTree;
import lombok.Cleanup;

import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;

/**
 *
 * PER = Prioritized Experience Replay
 * Reference:
 *  <a href="https://arxiv.org/abs/1511.05952">Prioritized Experience Replay</a>
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py">PER Python implementation</a>
 */
public class PrioritizedReplayBuffer implements IReplayBuffer<PrioritizedExperience> {

    public static final float DEFAULT_ALPHA = 0.6f;
    public static final float DEFAULT_BETA = 0.4f;

    private final PrioritizedExperience[] experiences;
    private final NDManager manager;
    private final int capacity;
    private final Device device;
    private int size;
    private int pos;
    private float maxPriority;
    // 0 = uniform distribution, 1 = full priority
    private float alpha;
    private final SumSegmentTree sumSegmentTree;
    private final MinSegmentTree minSegmentTree;

    public PrioritizedReplayBuffer(int capacity, NDManager manager) {
        this(capacity, DEFAULT_ALPHA, manager, Device.cpu());
    }

    public PrioritizedReplayBuffer(int capacity,
                                   float alpha,
                                   NDManager manager,
                                   Device device) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.experiences = new PrioritizedExperience[capacity];
        this.manager = requireNonNull(manager);
        this.device = requireNonNull(device);
        this.sumSegmentTree = new SumSegmentTree(this.capacity);
        this.minSegmentTree = new MinSegmentTree(this.capacity);
        this.pos = 0;
        this.alpha = alpha;
        this.maxPriority = 1f;
    }

    @Override
    public void store(PrioritizedExperience exp) {

        exp.state().attach(manager);
        exp.nextState().attach(manager);

        if (size < capacity) ++size;
        var oldExp = experiences[pos];
        experiences[pos] = exp;
        float defaultPriority = (float) Math.pow(maxPriority, alpha);
        sumSegmentTree.update(pos, defaultPriority);
        minSegmentTree.update(pos, defaultPriority);
        if (oldExp != null) oldExp.close();
        pos = (pos + 1) % experiences.length;
    }

    @Override
    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    /**
     * Approximation to the PER formula of priority sampling of experiences:
     * \( P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}} \)
     * @param batchSize batch size
     * @return array of prioritized experiences
     */
    protected PrioritizedExperience[] prioritizedSamples(int batchSize) {
        var batch = new PrioritizedExperience[batchSize];
        float segment = sumSegmentTree.sum() / batchSize;
        for (int i = 0; i < batchSize; ++i) {
            float lower = i * segment;
            float upper = (i + 1) * segment;
            int sampledIdx = sumSegmentTree.sampleIndexByValueInRange(lower, upper);
            batch[i] = experiences[sampledIdx];
        }

        return batch;
    }

    @Override
    public VecExperience sample(int batchSize) {
        return sample(batchSize, DEFAULT_BETA);
    }

    public VecExperience sample(int batchSize, float beta) {
        if (!enough(batchSize)) return null;

        @Cleanup var sub = manager.newSubManager();
        var batch = prioritizedSamples(batchSize);
        var states = toList(batch, exp -> {
            exp.state().tempAttach(sub);
            return exp.state().expandDims(0);
        }).toDevice(device, false);
        var actions = sub.create(stream(batch).mapToLong(exp -> exp.actionAs(Long.class)).toArray())
                .expandDims(1).toDevice(device, false);
        var rewards = sub.create(stream(batch).mapToDouble(PrioritizedExperience::reward).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);
        var nextStates = toList(batch, exp -> {
            exp.nextState().tempAttach(sub);
            return exp.nextState().expandDims(0);
        }).toDevice(device, false);
        var dones = sub.create(stream(batch).mapToDouble(exp -> exp.done() ? 1 : 0).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);
        var weights = calculateWeights(sub, batch, beta);

        return new VecExperience(
                sub.ret(states),
                sub.ret(actions),
                sub.ret(rewards),
                sub.ret(nextStates),
                sub.ret(dones),
                sub.ret(weights)
        );
    }

    /**
     * Calculates the Importance Sampling (IS) weights for the prioritized batch to correct
     * the bias introduced by non-uniform sampling.
     *
     * <p>The weight \( w_i \) for an experience \( i \) is defined as:
     * \[ w_i = \left( \frac{1}{N \cdot P(i)} \right)^{\beta} \]
     *
     * <p>To ensure stability, these weights are normalized by the maximum weight \( w_{max} \)
     * in the current batch:
     * \[ \hat{w}_i = \frac{w_i}{w_{max}} = \frac{(N \cdot P(i))^{-\beta}}{(N \cdot P_{min})^{-\beta}} \]
     *
     * @param sub   The {@link NDManager} to manage memory allocation for the resulting array.
     * @param batch The array of {@link PrioritizedExperience} sampled from the replay buffer.
     * @param beta  The degree of importance sampling correction (annealed from initial to 1.0).
     * @return An {@link NDArray} of shape (batchSize, 1) containing the normalized weights.
     */
    private NDArray calculateWeights(NDManager sub, PrioritizedExperience[] batch, float beta) {
        float sum = sumSegmentTree.sum();
        float pMin = minSegmentTree.min() / sum;
        int count = minSegmentTree.size();
        float maxISWeight = (float) Math.pow(count * pMin, -beta);
        return sub.create(stream(batch).mapToDouble(exp -> {
                    float pSample = exp.priority() / sum;
                    return Math.pow(count * pSample, -beta)/maxISWeight;
                })
                .toArray())
                .toType(DataType.FLOAT32, false)
                .expandDims(1)
                .toDevice(device, false);
    }

    public void updatePriorities(int[] idxs, NDArray priorities) {
        // TODO
    }

    /**
     * Immutable capacity, not the same as size
     * @return capacity
     */
    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public int size() {
        return size;
    }

    int pos() {
        return pos;
    }

    public record VecExperience(NDArray states,
                                NDArray actions,
                                NDArray rewards,
                                NDArray nextStates,
                                NDArray dones,
                                NDArray weights) implements IVecExperience {

    }

    @Override
    public void close() {
        for (var exp : experiences) {
            if (exp != null) exp.close();
        }
    }
}
