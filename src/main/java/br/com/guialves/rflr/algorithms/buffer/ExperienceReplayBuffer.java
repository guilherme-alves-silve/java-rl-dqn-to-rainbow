package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToFloat32;
import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToLong;
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
public class ExperienceReplayBuffer implements IReplayBuffer {

    private final Experience[] experiences;
    private final NDManager subManager;
    private final ExperienceSampler sampler;
    private final int capacity;
    private int size;
    private int pos;

    public ExperienceReplayBuffer(int capacity, NDManager manager) {
        this(capacity, manager, new ExperienceSampler());
    }

    public ExperienceReplayBuffer(int capacity,
                                  NDManager parent,
                                  ExperienceSampler sampler) {
        if (capacity <= 0) throw new IllegalArgumentException("Invalid capacity " + capacity + ": Must be greater than 0!");
        this.capacity = capacity;
        this.experiences = new Experience[capacity];
        this.subManager = subMgr(parent, getClass());
        this.sampler = requireNonNull(sampler);
        this.pos = 0;
    }

    @Override
    public void store(Experience exp) {

        exp.state().attach(subManager);
        exp.nextState().attach(subManager);

        if (size < capacity) ++size;
        var oldExp = experiences[pos];
        experiences[pos] = exp;
        setName(exp.state(), "buffer-state");
        setName(exp.nextState(), "buffer-next-state");
        if (oldExp != null) oldExp.close();
        pos = (pos + 1) % experiences.length;
    }

    @Override
    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    /**
     * @param batchSize The number of samples used to generate the mini-batch.
     * @return VecExperience record respecting this order (s, a, r, s', done)
     */
    @Override
    public VecExperience sample(int batchSize) {
        if (!enough(batchSize)) return null;

        var sub = subMgr(subManager, "buffer-sample");
        var batch = sampler.sample(experiences, batchSize, size, false);

        var states = newAttachedList(sub, batch, exp -> exp.state().duplicate());
        var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
        var rewards = djlMapToFloat32(sub, batch, Experience::reward);
        var nextStates = newAttachedList(sub, batch, exp -> exp.nextState().duplicate());
        var dones = djlMapToFloat32(sub, batch, exp -> exp.done() ? 1 : 0);

        return new VecExperience(
                sub,
                states,
                actions,
                rewards,
                nextStates,
                dones
        );
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
                                NDArray dones) implements IVecExperience {
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
