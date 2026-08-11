package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import br.com.guialves.rflr.datastructure.MinSegmentTree;
import br.com.guialves.rflr.datastructure.SumSegmentTree;
import lombok.NonNull;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.release;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static br.com.guialves.rflr.djlutils.DJLUtils.*;
import static java.util.Arrays.stream;

/**
 * Prioritized Experience Replay buffer with PER sampling and importance sampling weights.
 *
 * <p>Implements the core PER algorithm where experiences are sampled with probability
 * proportional to their TD-error priority (p^α), and bias is corrected via importance
 * sampling weights.</p>
 *
 * <p>Uses segment trees for O(log N) priority updates and O(log N) sampling.</p>
 * PER = Prioritized Experience Replay
 * Reference:
 *  @see <a href="https://arxiv.org/abs/1511.05952">Prioritized Experience Replay</a>
 *  @see <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py">PER Python implementation</a>
 */
public class PrioritizedReplayBuffer implements IReplayBuffer {

    public static final float DEFAULT_BETA = 0.4f;
    private static final float MIN_DELTA = 0.000_000_001f;

    private final Experience[] experiences;
    private final NDManager subManager;
    private final int capacity;
    private final Device device;
    private int size;
    private int pos;
    private float maxPriority;
    // 0 = uniform distribution, 1 = full priority
    private final float alpha;
    final SumSegmentTree sumSegmentTree;
    final MinSegmentTree minSegmentTree;

    public PrioritizedReplayBuffer(int capacity,
                                   float alpha,
                                   @NonNull NDManager parent) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.experiences = new Experience[capacity];
        this.subManager = subMgr(parent, getClass());
        this.device = parent.getDevice();
        this.sumSegmentTree = new SumSegmentTree(this.capacity);
        this.minSegmentTree = new MinSegmentTree(this.capacity);
        this.pos = 0;
        this.alpha = alpha;
        this.maxPriority = 1f;
    }

    @Override
    public void store(Experience exp) {

        exp.state().attach(subManager);
        exp.nextState().attach(subManager);

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

    private Experience[] buildPrioritizedSamples(int[] bufferIndexes) {
        var batch = new Experience[bufferIndexes.length];
        for (int i = 0; i < batch.length; ++i) {
            batch[i] = experiences[bufferIndexes[i]];
        }

        return batch;
    }

    /**
     * Approximation to the PER formula of priority sampling of experiences:
     * \( P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}} \)
     * @param batchSize batch size
     * @return array of the index of prioritized experiences
     */
    protected int[] prioritizedIndexSamples(int batchSize) {
        var batchIndexes = new int[batchSize];
        float segment = sumSegmentTree.sum() / batchSize;
        for (int i = 0; i < batchSize; ++i) {
            float lower = i * segment;
            float upper = (i + 1) * segment;
            int sampledIdx = sumSegmentTree.sampleIndexByValueInRange(lower, upper);
            batchIndexes[i] = sampledIdx;
        }

        return batchIndexes;
    }

    @Override
    public VecExperience sample(int batchSize) {
        return sample(batchSize, DEFAULT_BETA);
    }

    public VecExperience sample(int batchSize, float beta) {
        if (!enough(batchSize)) return null;

        var sub = subMgr(subManager, "prioritized-sample");
        var bufferIndexes = prioritizedIndexSamples(batchSize);
        var batch = buildPrioritizedSamples(bufferIndexes);

        var states = newAttachedList(sub, batch, exp -> exp.state().duplicate());
        var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
        var rewards = djlMapToFloat32(sub, batch, Experience::reward);
        var nextStates = newAttachedList(sub, batch, exp -> exp.nextState().duplicate());
        var dones = djlMapToFloat32(sub, batch, exp -> exp.done() ? 1 : 0);
        var weights = calculateWeights(sub, bufferIndexes, beta);

        return new VecExperience(
                sub,
                states,
                actions,
                rewards,
                nextStates,
                dones,
                weights,
                bufferIndexes
        );
    }

    /**
     * Calculates the Importance Sampling (IS) weights for the prioritized batch to correct
     * the bias introduced by non-uniform sampling.
     *
     * <p>The weight \( w^{\tiny (IS)}_i \) for an experience \( i \) is defined as:
     * \[ w^{\tiny (IS)}_i = \left(N \cdot P(i) \right)^{-\beta} \]
     *
     * <p>To ensure stability, these weights are normalized by the maximum weight \( w_{max} \)
     * in the current batch:
     * \[ \overline{w}^{\tiny (IS)}_i = \frac{w^{\tiny (IS)}_i}{w^{\tiny (IS)}_{max}} = \frac{(N \cdot P(i))^{-\beta}}{(N \cdot P_{min})^{-\beta}} \]
     *
     * @param sub           The {@link NDManager} to manage memory allocation for the resulting array.
     * @param bufferIndexes An array of integer indices referencing the experiences sampled
     *                      from the replay buffer. Each index corresponds to a position
     *                      in the underlying segment trees.
     * @param beta          The degree of importance sampling correction (annealed from initial to 1.0).
     * @return An {@link NDArray} of shape (batchSize, 1) containing the normalized weights.
     */
    private NDArray calculateWeights(NDManager sub, int[] bufferIndexes, float beta) {
        float sum = sumSegmentTree.sum();
        float pMin = Math.max(minSegmentTree.min() / sum, MIN_DELTA);
        int count = minSegmentTree.size();
        float maxISWeight = (float) Math.pow(count * pMin, -beta);
        return sub.create(stream(bufferIndexes)
                .mapToDouble(idx -> {
                    float pSample = sumSegmentTree.get(idx) / sum;
                    return Math.pow(count * pSample, -beta) / maxISWeight;
                })
                .toArray())
                .toType(DataType.FLOAT32, false)
                .expandDims(1)
                .toDevice(device, false);
    }

    /**
     * Updates priorities for given indices using priority transformation p^α.
     *
     * @param bufferIndexes      indices to update
     * @param priorities raw priority values (must match idxs length)
     * @throws IllegalArgumentException if lengths differ or any priority ≤ 0
     */
    public void updatePriorities(int[] bufferIndexes, NDArray priorities) {
        if (bufferIndexes.length != priorities.size()) throw new IllegalArgumentException("Invalid length!");
        float tempMaxPriority = this.maxPriority;
        // TODO: Create test for memory leak
        var rawPriorities = toFloatArray(priorities);
        for (int i = 0; i < bufferIndexes.length; ++i) {
            int experienceIdx = bufferIndexes[i];
            if (experienceIdx < 0 || experienceIdx >= experiences.length) {
                throw new ArrayIndexOutOfBoundsException("Invalid experienceIdx: " + experienceIdx);
            }

            float priority = (float) Math.pow(rawPriorities[i], this.alpha);
            if (priority <= 0) {
                throw new IllegalArgumentException("priority must be greater than 0!");
            }

            this.sumSegmentTree.update(experienceIdx, priority);
            this.minSegmentTree.update(experienceIdx, priority);
            tempMaxPriority = Math.max(tempMaxPriority, priority);
        }

        this.maxPriority = tempMaxPriority;
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

    public record VecExperience(NDManager sub,
                                NDArray states,
                                NDArray actions,
                                NDArray rewards,
                                NDArray nextStates,
                                NDArray dones,
                                NDArray weights,
                                int[] bufferIndexes) implements IVecExperience {

        @Override
        public void close() {
            sub.close();
        }
    }

    @Override
    public boolean isOpen() {
        return subManager.isOpen();
    }

    @Override
    public void close() {
        release(experiences);
        subManager.close();
    }
}
