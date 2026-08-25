package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import static br.com.guialves.rflr.djlutils.DJLUtils.*;

/**
 * Categorical NLL loss with PER importance sampling weights.
 *
 * <p>Weights must be set via {@link #normISWeights(NDArray)} before each forward pass.
 * After evaluation, weights are cleared to prevent stale usage.
 *
 * <p>IMPORTANT: Input {@code distLogits} MUST be raw logits.
 * The loss internally applies log-softmax transformation.
 *
 * <p>Loss formula: {@code L = -Σ wis_j Σ m_i * log(softmax(p_i(s,a,θ)))}
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
     * L = -Σ wis_j Σ m_i * log(softmax(p_i(s, a, θ)))
     *
     * @param massDist   the Bellman projected values of target-network
     * @param distLogits the raw logits of online-network (log-softmax applied internally)
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

        // (batch) - We need to extract the loss per element to update sum/min segment-tree
        return loss.squeeze();
    }
}
