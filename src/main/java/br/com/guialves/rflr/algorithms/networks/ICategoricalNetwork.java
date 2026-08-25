package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import lombok.Cleanup;

import java.util.function.UnaryOperator;

public interface ICategoricalNetwork extends IDeepQNetwork {

    /**
     * Applies softmax transformation to action distributions over atom supports.
     *
     * <p>For each action, the distribution over atoms is converted to probabilities
     * using softmax.
     *
     * <p>The output shape is {@code (batch, actions, atoms)}, with softmax applied along the atoms
     * dimension (index 2).</p>
     *
     * <p>Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/softmax-regression-djl.html">
     * Softmax Regression — DJL</a></p>
     *
     * @param inputs the input logits for each action-atom pair
     * @return the softmax transformed probabilities in the specified shape
     */
    NDArray forwardDist(NDList inputs, final UnaryOperator<NDArray> block);

    /**
     * Forward pass returning raw logits of shape {@code (batch, actions, atoms)}.
     *
     * @param input the network input tensor
     * @param block transformation applied after reshaping (e.g., {@code nd -> nd.logSoftmax(LAST_AXIS)})
     * @return logits tensor of shape {@code (batch, actions, atoms)}
     */
    NDArray forwardLogits(NDArray input, final UnaryOperator<NDArray> block);

    /**
     * Computes the Q-value for each (batch, action) pair as the expectation of the
     * categorical return distribution over the atom support {@code z}:
     * {@code Q(s, a) = sum_i z_i * p_i(s, a)}. Works for both online ({@code p(s, a)})
     * and target-network ({@code p(s', a)}) distributions.
     *
     * @param distribution categorical distribution over atoms, shape {@code (batch, actions, atoms)}
     * @return Q-values, shape {@code (batch, actions)}
     * @throws IllegalStateException if {@code distribution} is not rank 3
     */
    NDArray qValuesFromDist(NDArray distribution);

    NDArray projectBellman(final NDArray probNextDist,
                           final NDArray rewards,
                           final NDArray dones,
                           final float gamma);

    @Override
    default NDList forward(NDList input) {
        @Cleanup var dist = forwardDist(input);
        return new NDList(qValuesFromDist(dist));
    }

    @Override
    default NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    default NDArray forwardDist(NDList inputs) {
        return forwardDist(inputs, UnaryOperator.identity());
    }

    default NDArray forwardDist(NDArray input, final UnaryOperator<NDArray> block) {
        return forwardDist(new NDList(input), block);
    }
}
