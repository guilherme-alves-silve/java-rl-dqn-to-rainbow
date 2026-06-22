package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.function.Function;

import static java.util.Arrays.stream;

/**
 * Vectorized experience replay buffer
 * Sources:
 * <a href="https://d2l.djl.ai/chapter_preliminaries/ndarray.html">DLJ - 2.1. Data Manipulation</a>
 * <a href="https://neuralpalace.substack.com/p/how-to-never-forget-deep-q-networks">How to Never Forget Deep
 *  Q-Networks: Memory Palaces Meet Reinforcement Learning</a>
 * <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01_dqn.py">rainbow-is-all-you-need/01_dqn.py</a>
 */
public class ExperienceReplayBuffer {

    private final Experience[] experiences;
    private final NDManager manager;
    private final ExperienceSampler sampler;
    private final int capacity;
    private int size;
    private int pos;

    public ExperienceReplayBuffer(int capacity, NDManager manager) {
        this(capacity, manager, new ExperienceSampler());
    }

    public ExperienceReplayBuffer(int capacity, NDManager manager, ExperienceSampler sampler) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.experiences = new Experience[capacity];
        this.manager = manager;
        this.sampler = sampler;
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
        var states = toNDList(batch, Experience::state);
        var actions = manager.create(stream(batch).mapToInt(exp -> exp.actionAs(int.class)).toArray());
        var rewards = manager.create(stream(batch).mapToDouble(Experience::reward).toArray());
        var nextStates = toNDList(batch, Experience::nextState);
        var dones = manager.create(stream(batch).mapToInt(exp -> exp.done()? 1 : 0).toArray());

        return new VecExperience(states, actions, rewards, nextStates, dones);
    }

    protected NDList toNDList(Experience[] batch, Function<Experience, NDArray> mapper) {
        return stream(batch)
                .map(mapper)
                .collect(() -> new NDList(batch.length), NDList::add, NDList::addAll);
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

    public record VecExperience(NDList states,
                                NDArray actions,
                                NDArray rewards,
                                NDList nextStates,
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
