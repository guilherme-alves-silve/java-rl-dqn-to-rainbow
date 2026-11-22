package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * The batch of a prioritized replay: the usual {@code (s, a, r, s', done)} plus the
 * importance-sampling weights and the buffer indexes needed to update the priorities.
 *
 * @param sub           the manager that owns the arrays of this batch
 * @param states        the states of the batch
 * @param actions       the actions of the batch
 * @param rewards       the rewards of the batch
 * @param nextStates    the next states of the batch
 * @param dones         the termination flags of the batch
 * @param weights       the normalized importance-sampling weights
 * @param bufferIndexes the index of each sample in the buffer that owns it
 */
public record PrioritizedVecExperience(NDManager sub,
                                       NDArray states,
                                       NDArray actions,
                                       NDArray rewards,
                                       NDArray nextStates,
                                       NDArray dones,
                                       NDArray weights,
                                       int[] bufferIndexes) implements IVecExperience {

    @Override
    public void close() {
        sub.close();
    }
}
