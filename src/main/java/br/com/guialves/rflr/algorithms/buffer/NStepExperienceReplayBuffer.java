package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

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
public class NStepExperienceReplayBuffer extends ExperienceReplayBuffer implements INStepBuffer {

    private final int nStep;
    private final float gamma;
    /**
     * One deque per environment. A vectorized environment steps all the
     * environments in lockstep, so the n-step window of one environment must not
     * be mixed with the window of another one.
     */
    private final Map<Integer, Deque<Experience>> nStepDeques;

    public NStepExperienceReplayBuffer(int capacity,
                                       int nStep,
                                       float gamma,
                                       NDManager parent) {
        this(capacity, nStep, gamma, parent, new ExperienceSampler());
    }

    public NStepExperienceReplayBuffer(int capacity,
                                       int nStep,
                                       float gamma,
                                       NDManager parent,
                                       ExperienceSampler sampler) {
        super(capacity, parent, sampler);
        if (nStep < 1) throw new IllegalArgumentException("nStep must at least 1: " + nStep);
        if (gamma <= 0 || gamma > 1) throw new IllegalArgumentException("gamma must be between range (0, 1]: " + gamma);
        this.nStep = nStep;
        this.gamma = gamma;
        this.nStepDeques = new HashMap<>();
    }

    /**
     * Below you can see the original formula, used as reference:
     * \( y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max\limits_a Q_{target}(s_{t+n}, a') \cdot (1 - \text{done}) \)
     * @param exp Actual experience
     */
    @Override
    public void store(Experience exp) {
        keepAlive(exp);
        var nStepDeque = nStepDeques.computeIfAbsent(exp.envNumber(),
                _ -> new ArrayDeque<>(nStep));
        nStepDeque.addLast(exp);

        if (!exp.done()) {
            if (nStepDeque.size() == nStep) {
                emitExperience(nStepDeque);
                nStepDeque.pollFirst().close();
            }
            return;
        }

        while (!nStepDeque.isEmpty()) {
            emitExperience(nStepDeque);
            nStepDeque.pollFirst().close();
        }
    }

    private void emitExperience(Deque<Experience> nStepDeque) {
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
        int envNumber = oldest.envNumber();
        var oldState = oldest.state();
        var oldAction = oldest.action();
        var newNextState = newest.nextState();
        var term = newest.term();
        var trunc = newest.trunc();

        var nStepExperience = Experience.copy(envNumber, oldState, oldAction,
                nStepReward, newNextState, term, trunc);

        super.store(nStepExperience);
    }

    public int nStep() {
        return nStep;
    }

    @Override
    public void close() {
        super.close();
        nStepDeques.values().forEach(nStepDeque -> nStepDeque.forEach(Experience::close));
        nStepDeques.clear();
    }
}
