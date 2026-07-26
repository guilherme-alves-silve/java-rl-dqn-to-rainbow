package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayDeque;
import java.util.Deque;

import static java.util.Objects.requireNonNull;

/**
 * N-Step Vectorized experience replay buffer
 * Sources:
 * <a href="https://d2l.djl.ai/chapter_preliminaries/ndarray.html">DLJ - 2.1. Data Manipulation</a>
 * <a href="https://neuralpalace.substack.com/p/how-to-never-forget-deep-q-networks">How to Never Forget Deep
 *  Q-Networks: Memory Palaces Meet Reinforcement Learning</a>
 * <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01_dqn.py">rainbow-is-all-you-need/01_dqn.py</a>
 */
@Slf4j
public class NStepExperienceReplayBuffer extends ExperienceReplayBuffer {

    private final int nStep;
    private final float gamma;
    private final Deque<Experience> nStepDeque;

    public NStepExperienceReplayBuffer(int nStep,
                                       float gamma,
                                       int capacity,
                                       NDManager manager) {
        this(nStep, gamma, capacity, manager, new ExperienceSampler());
    }

    public NStepExperienceReplayBuffer(int nStep,
                                       float gamma,
                                       int capacity,
                                       NDManager manager,
                                       ExperienceSampler sampler) {
        super(capacity, manager, sampler);
        if (nStep < 1) throw new IllegalArgumentException("nStep must at least 1!");
        if (gamma <= 0 || gamma > 1) throw new IllegalArgumentException("gamma must be between range (0, 1]!");
        this.nStep = nStep;
        this.gamma = gamma;
        this.nStepDeque = new ArrayDeque<>(nStep);
    }

    /**
     * Below you can see the original formula, used as reference:
     * \( y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max\limits_a Q_{target}(s_{t+n}, a') \cdot (1 - \text{done}) \)
     * @param exp Actual experience
     */
    @Override
    public void store(Experience exp) {

        nStepDeque.addLast(exp);
        if (nStepDeque.size() < nStep) return;

        double nStepReward = 0f;
        var oldest = nStepDeque.getFirst();
        int k = 0;
        Experience newest = null;
        for (var curr : nStepDeque) {
            nStepReward += Math.pow(gamma, k) * curr.reward();
            newest = curr;
            if (curr.done()) break;
            ++k;
        }
        requireNonNull(newest, "Cannot be null!");
        var oldState = oldest.state().duplicate();
        var oldAction = oldest.action().duplicate();
        var newNextState = newest.nextState().duplicate();
        var done = newest.done();

        var nStepExperience = new Experience(oldState, oldAction,
                nStepReward, newNextState, done);

        super.store(nStepExperience);
        nStepDeque.pollFirst().close();
    }

    public int nStep() {
        return nStep;
    }
}
