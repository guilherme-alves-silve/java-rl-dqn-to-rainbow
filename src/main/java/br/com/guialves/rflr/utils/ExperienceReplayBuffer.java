package br.com.guialves.rflr.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Function;

import static java.util.Arrays.stream;
import static java.util.Objects.requireNonNull;

/**
 * Vectorized experience replay buffer
 * Sources:
 * <a href="https://d2l.djl.ai/chapter_preliminaries/ndarray.html">DLJ - 2.1. Data Manipulation</a>
 * <a href="https://neuralpalace.substack.com/p/how-to-never-forget-deep-q-networks">How to Never Forget Deep
 *  Q-Networks: Memory Palaces Meet Reinforcement Learning</a>
 * <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01_dqn.py">rainbow-is-all-you-need/01_dqn.py</a>
 */
@Slf4j
public class ExperienceReplayBuffer {

    private final Experience[] experiences;
    private final NDManager manager;
    private final ExperienceSampler sampler;
    private final int capacity;
    private final Device device;
    private int size;
    private int pos;

    public ExperienceReplayBuffer(int capacity, NDManager manager) {
        this(capacity, manager, new ExperienceSampler(), Device.cpu());
    }

    public ExperienceReplayBuffer(int capacity, NDManager manager, ExperienceSampler sampler) {
        this(capacity, manager, sampler, Device.cpu());
    }

    public ExperienceReplayBuffer(int capacity,
                                  NDManager manager,
                                  ExperienceSampler sampler,
                                  Device device) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.experiences = new Experience[capacity];
        this.manager = requireNonNull(manager);
        this.sampler = requireNonNull(sampler);
        this.device = requireNonNull(device);
        this.pos = 0;
    }

    public void store(Experience exp) {
        exp.state().attach(manager);
        exp.nextState().attach(manager);

        if (size < capacity) ++size;
        var temp = experiences[pos];
        experiences[pos] = exp;
        if (temp != null) temp.close();
        pos = (pos + 1) % experiences.length;
    }

    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    /**
     * @param batchSize The number of samples used to generate the mini-batch.
     * @return VecExperience record respecting this order (s, a, r, s', done)
     */
    public VecExperience sample(int batchSize) {
        if (!enough(batchSize)) return null;

        var batch = sampler.sample(experiences, batchSize, size, false);
        var states = toList(batch, exp -> exp.state().expandDims(0))
                .toDevice(device, false);
        var actions = manager.create(stream(batch).mapToLong(exp -> exp.actionAs(Long.class)).toArray())
                .expandDims(1).toDevice(device, false);;
        var rewards = manager.create(stream(batch).mapToDouble(Experience::reward).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);
        var nextStates = toList(batch, exp -> exp.nextState().expandDims(0))
                .toDevice(device, false);;
        var dones = manager.create(stream(batch).mapToDouble(exp -> exp.done()? 1 : 0).toArray())
                .toType(DataType.FLOAT32, false).expandDims(1).toDevice(device, false);

        return new VecExperience(states, actions, rewards, nextStates, dones);
    }

    /**
     * Converts an array of Experiences into a batched NDArray using the provided mapper.
     *
     * <p>The mapper function must add a batch dimension using {@code expandDims(0)}
     * to preserve the batch structure. For example, if each Experience maps to
     * a {@code [3, 3]} NDArray, the mapper should return {@code [1, 3, 3]}.
     *
     * <p><b>Shape behavior:</b>
     * <ul>
     *   <li>With {@code expandDims(0)}: {@code [1, 3, 3]} → {@code [N, 3, 3]} ✓</li>
     *   <li>Without {@code expandDims(0)}: {@code [3, 3]} → {@code [N*3, 3]} ✗</li>
     * </ul>
     *
     * @param batch  the array of experiences to batch
     * @param mapper maps each experience to its NDArray representation
     * @return concatenated NDArray along axis 0
     */
    protected NDArray toList(Experience[] batch, Function<Experience, NDArray> mapper) {
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
                                NDArray dones) implements AutoCloseable {

        @Override
        public void close() {
            states.close();
            actions.close();
            rewards.close();
            nextStates.close();
            dones.close();
        }
    }
}
