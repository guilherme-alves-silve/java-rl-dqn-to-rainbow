package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

public class CategoricalCrossEntropyLoss extends Loss {

    private final int classAxis;

    public CategoricalCrossEntropyLoss(int classAxis) {
        super(CategoricalCrossEntropyLoss.class.getSimpleName());
        this.classAxis = classAxis;
    }

    @Override
    public NDArray evaluate(NDList massDist, NDList dist) {
        var pred = dist.singletonOrThrow();
        var lab = massDist.singletonOrThrow();
        var loss = pred.mul(lab).neg().sum(new int[] {classAxis}, true);
        return loss.mean();
    }
}
