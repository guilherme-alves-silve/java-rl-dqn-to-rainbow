package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;

/**
 * The capability of a Prioritized Experience Replay buffer.
 *
 * <p>The agents used to check {@code instanceof PrioritizedReplayBuffer}, which made
 * impossible to use any other implementation of the PER algorithm - for instance the
 * {@link NEnvReplayBuffer}, that holds one prioritized buffer per environment. This
 * interface describes what a prioritized buffer is able to do, and nothing else.</p>
 */
public interface IPrioritizedBuffer extends IReplayBuffer {

    public static final float MIN_PRIORITY = 0.000_000_001f;

    /**
     * Samples a mini-batch with the priorities taken into account.
     *
     * @param batchSize the number of samples
     * @param beta      the degree of the importance-sampling correction
     * @return the batch with the importance-sampling weights and the buffer indexes
     */
    PrioritizedVecExperience sample(int batchSize, float beta);

    /**
     * Updates the priorities of the given samples.
     *
     * @param bufferIndexes the index of each sample in the buffer that owns it
     * @param priorities    the new priorities, one per sample
     */
    void updatePriorities(int[] bufferIndexes, NDArray priorities);
}
