package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
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
    private final Shape stateShape;
    private int size;
    private int pos;

    public ExperienceReplayBuffer(int capacity, Shape stateShape, NDManager manager) {
        this(capacity, stateShape, manager, new ExperienceSampler());
    }

    public ExperienceReplayBuffer(int capacity,
                                  Shape stateShape,
                                  NDManager manager,
                                  ExperienceSampler sampler) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.stateShape = requireNonNull(stateShape);
        this.experiences = new Experience[capacity];
        this.manager = requireNonNull(manager);
        this.sampler = requireNonNull(sampler);
        this.pos = 0;
    }

    public void store(Experience exp) {
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

        var batch = sampler.sample(experiences, batchSize, false);
        var states = toList(batch, exp -> exp.state().expandDims(0));
        var actions = manager.create(stream(batch).mapToInt(exp -> exp.actionAs(int.class)).toArray());
        var rewards = manager.create(stream(batch).mapToDouble(Experience::reward).toArray());
        var nextStates = toList(batch, exp -> exp.nextState().expandDims(0));
        var dones = manager.create(stream(batch).mapToInt(exp -> exp.done()? 1 : 0).toArray());

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
                .peek(arr -> {
                    if (arr.getShape().dimension() == 2) {
                        log.warn("Mapper returned 2D array (shape: {}). " +
                                        "Consider using expandDims(0) for proper batching.",
                                arr.getShape());
                    }
                })
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
