package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToFloat32;
import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToLong;
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
public class ExperienceReplayBuffer implements IReplayBuffer<Experience> {

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

    public ExperienceReplayBuffer(int capacity, NDManager manager, Device device) {
        this(capacity, manager, new ExperienceSampler(), device);
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

    @Override
    public void store(Experience exp) {

        exp.state().attach(manager);
        exp.nextState().attach(manager);

        if (size < capacity) ++size;
        var oldExp = experiences[pos];
        experiences[pos] = exp;
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

        @Cleanup var sub = manager.newSubManager();
        var batch = sampler.sample(experiences, batchSize, size, false);

        var states = newAttachedList(sub, device, batch, IExperience::state);
        var actions = djlMapToLong(sub, device, batch, exp -> exp.actionAs(Long.class));
        var rewards = djlMapToFloat32(sub, device, batch, Experience::reward);
        var nextStates = newAttachedList(sub, device, batch, IExperience::nextState);
        var dones = djlMapToFloat32(sub, device, batch, exp -> exp.done() ? 1 : 0);

        return new VecExperience(
                sub.ret(states),
                sub.ret(actions),
                sub.ret(rewards),
                sub.ret(nextStates),
                sub.ret(dones)
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

    public record VecExperience(NDArray states,
                                NDArray actions,
                                NDArray rewards,
                                NDArray nextStates,
                                NDArray dones) implements IVecExperience {

    }

    @Override
    public void close() {
        for (var exp : experiences) {
            if (exp != null) exp.close();
        }
    }
}
