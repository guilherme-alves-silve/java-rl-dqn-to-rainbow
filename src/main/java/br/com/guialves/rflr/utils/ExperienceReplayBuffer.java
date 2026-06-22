package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Vectorized experience replay buffer
 */
public class ExperienceReplayBuffer {

    private final int size;
    private final Experience[] experiences;
    private final NDManager manager;
    private final ExperienceSampler sampler;
    private int pos;

    public ExperienceReplayBuffer(int size, NDManager manager) {
        if (size <= 0) throw new IllegalArgumentException("Invalid size " + size + ": Must be greater than 0!");
        this.size = size;
        this.experiences = new Experience[size];
        this.manager = manager;
        this.sampler = new ExperienceSampler();
        this.pos = 0;
    }

    public void store(Experience exp) {
        pos = (pos + 1) % experiences.length;
        experiences[pos] = exp;
    }

    public boolean enough(int batchSize) {
        return pos >= batchSize;
    }

    /**
     *
     * @param batchSize
     * @return VecExperience record respecting this order (s, a, r, s', done)
     */
    public VecExperience sample(int batchSize) {
        if (!enough(batchSize)) return null;

        var batch = sampler.sample(experiences, batchSize, false);
        var states = toNDList(batch, Experience::state);
        var actions = manager.create(Arrays.stream(batch).mapToInt(exp -> exp.actionAs(Integer.class)).toArray());
        var rewards = manager.create(Arrays.stream(batch).mapToDouble(Experience::reward).toArray());
        var nextStates = toNDList(batch, Experience::nextState);
        var dones = manager.create(Arrays.stream(batch).mapToInt(exp -> exp.done()? 1 : 0).toArray());

        return new VecExperience(states, actions, rewards, nextStates, dones);
    }

    protected NDList toNDList(Experience[] batch, Function<Experience, NDArray> mapper) {
        return Arrays.stream(batch)
                .map(mapper)
                .collect(() -> new NDList(batch.length), NDList::add, NDList::addAll);
    }

    public int size() {
        return size;
    }

    public record VecExperience(NDList states,
                                NDArray actions,
                                NDArray rewards,
                                NDList nextStates,
                                NDArray dones) {}
}
