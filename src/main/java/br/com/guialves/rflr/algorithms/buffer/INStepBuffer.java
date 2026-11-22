package br.com.guialves.rflr.algorithms.buffer;

/**
 * The capability of a replay buffer that accumulates the n-step return.
 *
 * <p>Same reason as {@link IPrioritizedBuffer}: the agents used to check
 * {@code instanceof NStepExperienceReplayBuffer} / {@code instanceof
 * NStepPrioritizedReplayBuffer}, which prevented the use of any other implementation.</p>
 */
public interface INStepBuffer extends IReplayBuffer {

    /**
     * @return the number of steps accumulated to compute the return
     */
    int nStep();
}
