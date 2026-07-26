package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayDeque;
import java.util.Deque;

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
public class NStepExperienceReplayBuffer extends ExperienceReplayBuffer {

    private final int nStep;
    private final Deque<Experience> nStepHolder;

    public NStepExperienceReplayBuffer(int capacity, int nStep, NDManager manager) {
        this(capacity, nStep, manager, new ExperienceSampler());
    }

    public NStepExperienceReplayBuffer(int capacity,
                                       int nStep,
                                       NDManager manager,
                                       ExperienceSampler sampler) {
        super(capacity, manager, sampler);
        this.nStep = nStep;
        this.nStepHolder = new ArrayDeque<>(nStep);
    }

    @Override
    public void store(Experience exp) {

        nStepHolder.add(exp);
        if (nStepHolder.size() < nStep) return;
        

        super.store(exp);
    }

    public int nStep() {
        return nStep;
    }

    /**
     * @param batchSize The number of samples used to generate the mini-batch.
     * @return VecExperience record respecting this order (s, a, r, s', done)
     */
    @Override
    public VecExperience sample(int batchSize) {
        return super.sample(batchSize);
    }
}
