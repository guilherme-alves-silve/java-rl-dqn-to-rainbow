package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import static br.com.guialves.rflr.djlutils.DJLUtils.*;

/**
 * Categorical Negative Log-Likelihood PER Loss with Importance Sampling weights for bias correction, for the Categorical DQN,
 *
 * <p>Weights must be set via {@link #normISWeights(NDArray)} before each forward pass.
 * After evaluation, weights are cleared to prevent stale usage.
 *
 * <p>
 * IMPORTANT: The input 'dist' MUST be log-softmax probabilities.
 * If you pass raw logits or softmax probabilities, the gradients will be wrong.
 */
public class CategoricalNLLPERLoss extends Loss {

    private final Reduction reduction;
    private NDArray normISWeights;

    public enum Reduction {
        MEAN, NONE
    }

    /**
     * Importance Sampling Weights used in the PER algorithm
     */
    public CategoricalNLLPERLoss(Reduction reduction) {
        super(CategoricalNLLPERLoss.class.getSimpleName());
        this.reduction = reduction;
    }

    public static CategoricalNLLPERLoss noneReduction() {
        return new CategoricalNLLPERLoss(Reduction.NONE);
    }

    /**
     * Sets the normalized importance sampling weights for the current batch.
     *
     * @param normISWeights IS weights with shape (batchSize,) or (batchSize, 1)
     */
    public CategoricalNLLPERLoss normISWeights(NDArray normISWeights) {
        if (normISWeights.getShape().dimension() != 3) {
            normISWeights = normISWeights.reshape(N_BATCH, 1, 1);
        }

        this.normISWeights = normISWeights;
        return this;
    }

    /**
     * L = -\sum wis_j \sum m_i * p_i(s, a, \theta)
     * @param massDist the Bellman projected values of target-network
     * @param distLogits the predicted values of online-network
     * @return Loss
     */
    @Override
    public NDArray evaluate(NDList massDist, NDList distLogits) {
        if (null == normISWeights) {
            throw new IllegalStateException("You must set the normISWeights before training!");
        }

        // (batch, 1, atoms)
        var predLogits = distLogits.singletonOrThrow();
        // (batch, 1, atoms)
        var lab = massDist.singletonOrThrow();
        // (batch, 1, 1) - pred must be logits
        var loss = normISWeights.mul(predLogits.logSoftmax(LAST_AXIS)
                        .mul(lab)
                        .sum(LAST_AXIS_ARR, KEEP_DIMS))
                        .neg();
        this.normISWeights = null;
        if (reduction.equals(Reduction.MEAN)) {
            return loss.mean();
        }

        // (batch, 1) - We need to extract the loss per element to update sum/min segment-tree
        return loss.squeeze(AXIS_1);
    }
}
