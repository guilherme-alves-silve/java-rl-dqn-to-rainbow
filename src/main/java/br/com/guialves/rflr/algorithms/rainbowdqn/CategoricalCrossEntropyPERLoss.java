package br.com.guialves.rflr.algorithms.rainbowdqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;
import br.com.guialves.rflr.djlutils.DJLUtils;

import static br.com.guialves.rflr.djlutils.DJLUtils.KEEP_DIMS;
import static br.com.guialves.rflr.djlutils.DJLUtils.LAST_AXIS_ARR;

/**
 * PER Loss with Importance Sampling weights for bias correction, for the Categorical DQN,
 *
 * <p>Weights must be set via {@link #normISWeights(NDArray)} before each forward pass.
 * After evaluation, weights are cleared to prevent stale usage.
 */
public class CategoricalCrossEntropyPERLoss extends Loss {

    private static final float HALF_WEIGHT = 0.5f;
    private final int atoms;
    private final Reduction reduction;
    private NDArray normISWeights;

    public enum Reduction {
        MEAN, NONE
    }

    /**
     * Importance Sampling Weights used in the PER algorithm
     */
    public CategoricalCrossEntropyPERLoss(int atoms, Reduction reduction) {
        super(CategoricalCrossEntropyPERLoss.class.getSimpleName());
        this.atoms = atoms;
        this.reduction = reduction;
    }

    public static CategoricalCrossEntropyPERLoss noneReduction(int atoms) {
        return new CategoricalCrossEntropyPERLoss(atoms, Reduction.NONE);
    }



    /**
     * Sets the normalized importance sampling weights for the current batch.
     *
     * @param normISWeights IS weights with shape (batchSize,) or (batchSize, 1)
     */
    public CategoricalCrossEntropyPERLoss normISWeights(NDArray normISWeights) {
        if (normISWeights.getShape().dimension() == 1) {
            normISWeights = normISWeights.reshape(DJLUtils.N_BATCH, 1, atoms);
        }

        this.normISWeights = normISWeights;
        return this;
    }

    /**
     * L = -\sum wis_j \sum m_i * p_i(s, a, \theta)
     * @param massDist the Bellman projected values of target-network
     * @param dist the predicted values of online-network
     * @return Loss
     */
    @Override
    public NDArray evaluate(NDList massDist, NDList dist) {
        if (null == normISWeights) {
            throw new IllegalStateException("You must set the normISWeights before training!");
        }

        // (batch, 1, atoms)
        var pred = dist.singletonOrThrow();
        // (batch, 1, atoms)
        var lab = massDist.singletonOrThrow();
        // (batch, 1, 1) - pred must be log softmax
        var loss = normISWeights.sum()
                .mul(pred.mul(lab)
                .sum(LAST_AXIS_ARR, KEEP_DIMS)).neg();
        this.normISWeights = null;
        if (reduction.equals(Reduction.MEAN)) {
            return loss.mean();
        }

        return loss;
    }
}
