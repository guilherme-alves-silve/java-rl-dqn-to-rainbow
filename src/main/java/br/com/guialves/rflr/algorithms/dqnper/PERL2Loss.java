package br.com.guialves.rflr.algorithms.dqnper;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import static br.com.guialves.rflr.djlutils.DJLUtils.N_BATCH;

/**
 * PER Loss with Importance Sampling weights for bias correction.
 *
 * <p>Implements the weighted L2 loss: L = ½ * w_i * (pred - label)²
 * where w_i are the importance sampling weights from PER.
 *
 * <p>Weights must be set via {@link #normISWeights(NDArray)} before each forward pass.
 * After evaluation, weights are cleared to prevent stale usage.
 */
public class PERL2Loss extends Loss {

    private static final float HALF_WEIGHT = 0.5f;
    private final Reduction reduction;
    private NDArray normISWeights;

    public enum Reduction {
        MEAN, NONE
    }

    /**
     * Importance Sampling Weights used in the PER algorithm
     */
    public PERL2Loss(Reduction reduction) {
        super(PERL2Loss.class.getSimpleName());
        this.reduction = reduction;
    }

    public static PERL2Loss noneReduction() {
        return new PERL2Loss(Reduction.NONE);
    }

    /**
     * Sets the normalized importance sampling weights for the current batch.
     *
     * @param normISWeights IS weights with shape (batchSize,) or (batchSize, 1)
     */
    public PERL2Loss normISWeights(NDArray normISWeights) {
        if (normISWeights.getShape().dimension() != 2) {
            normISWeights = normISWeights.reshape(N_BATCH, 1);
        }

        this.normISWeights = normISWeights;
        return this;
    }

    @Override
    public NDArray evaluate(NDList label, NDList prediction) {
        if (null == normISWeights) {
            throw new IllegalStateException("You must set the normISWeights before training!");
        }
        var pred = prediction.singletonOrThrow();
        var labelReshaped = label.singletonOrThrow().reshape(pred.getShape());
        var timeDifferenceError = labelReshaped.sub(pred).square();
        var weightedError = timeDifferenceError.mul(normISWeights);
        var loss = weightedError.mul(HALF_WEIGHT);
        this.normISWeights = null;
        if (reduction.equals(Reduction.MEAN)) {
            return loss.mean();
        }

        // (batch, 1) - We need to extract the loss per element to update sum/min segment-tree
        return loss;
    }
}
