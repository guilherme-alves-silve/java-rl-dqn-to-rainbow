package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import br.com.guialves.rflr.datastructure.MinSegmentTree;
import br.com.guialves.rflr.datastructure.SumSegmentTree;
import lombok.Cleanup;

import java.util.function.Function;

import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;

/**
 *
 * PER = Prioritized Experience Replay
 * Reference:
 *  <a href="https://arxiv.org/abs/1511.05952">Prioritized Experience Replay</a>
 */
public class PrioritizedReplayBuffer implements AutoCloseable {

    private final PrioritizedExperience[] experiences;
    private final NDManager manager;
    private final int capacity;
    private final Device device;
    private int size;
    private int pos;
    private final SumSegmentTree sumSegmentTree;
    private final MinSegmentTree minSegmentTree;

    public PrioritizedReplayBuffer(int capacity, NDManager manager) {
        this(capacity, manager, Device.cpu());
    }

    public PrioritizedReplayBuffer(int capacity,
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
    }

    public void store(PrioritizedExperience exp) {

        exp.state().attach(manager);
        exp.nextState().attach(manager);

        if (size < capacity) ++size;
        var oldExp = experiences[pos];
        experiences[pos] = exp;
        sumSegmentTree.update(pos, exp.priority());
        minSegmentTree.update(pos, exp.priority());
        if (oldExp != null) oldExp.close();
        pos = (pos + 1) % experiences.length;
    }

    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    private PrioritizedExperience[] prioritizedSamples(int batchSize) {
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

    public IVecExperience sample(int batchSize) {
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
        })
                .toDevice(device, false);
        var dones = sub.create(stream(batch).mapToDouble(exp -> exp.done() ? 1 : 0).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);
        var priorities = sub.create(stream(batch).mapToDouble(PrioritizedExperience::priority).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);

        return new VecExperience(
                sub.ret(states),
                sub.ret(actions),
                sub.ret(rewards),
                sub.ret(nextStates),
                sub.ret(dones),
                sub.ret(priorities)
        );
    }

    protected NDArray toList(PrioritizedExperience[] batch,
                             Function<PrioritizedExperience, NDArray> mapper) {
        var arrays = stream(batch)
                .map(mapper)
                .collect(() -> new NDList(batch.length), NDList::add, NDList::addAll);
        return NDArrays.concat(arrays, 0);
    }

    /**
     * Immutable capacity, not the same as size
     * @return capacity
     */
    public int capacity() {
        return capacity;
    }

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
                                NDArray priorities) implements IVecExperience {

    }

    @Override
    public void close() {
        for (var exp : experiences) {
            exp.close();
        }
    }
}
