package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

public class CategoricalCrossEntropyLoss extends Loss {

    private static final int[] AXIS_1_ARR = new int[] {1};

    public CategoricalCrossEntropyLoss() {
        super(CategoricalCrossEntropyLoss.class.getSimpleName());
    }

    @Override
    public NDArray evaluate(NDList massDist, NDList dist) {
        var pred = dist.singletonOrThrow();
        var lab = massDist.singletonOrThrow();
        // pred must be log softmax
        var loss = pred.mul(lab).neg().sum(AXIS_1_ARR, true);
        return loss.mean();
    }
}
