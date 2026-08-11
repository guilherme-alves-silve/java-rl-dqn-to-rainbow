package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import static br.com.guialves.rflr.djlutils.DJLUtils.KEEP_DIMS;
import static br.com.guialves.rflr.djlutils.DJLUtils.LAST_AXIS_ARR;

public class CategoricalCrossEntropyLoss extends Loss {

    public CategoricalCrossEntropyLoss() {
        super(CategoricalCrossEntropyLoss.class.getSimpleName());
    }

    @Override
    public NDArray evaluate(NDList massDist, NDList dist) {
        // (batch, 1, atoms)
        var pred = dist.singletonOrThrow();
        // (batch, 1, atoms)
        var lab = massDist.singletonOrThrow();
        // (batch, 1, 1) - pred must be log softmax
        var loss = pred.mul(lab).neg().sum(LAST_AXIS_ARR, KEEP_DIMS);
        // () -> scalar
        return loss.mean();
    }
}
