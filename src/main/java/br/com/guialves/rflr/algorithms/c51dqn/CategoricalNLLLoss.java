package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import static br.com.guialves.rflr.djlutils.DJLUtils.*;

public class CategoricalNLLLoss extends Loss {

    public CategoricalNLLLoss() {
        super(CategoricalNLLLoss.class.getSimpleName());
    }

    @Override
    public NDArray evaluate(NDList massDist, NDList distLogits) {
        // (batch, 1, atoms)
        var predLogits = distLogits.singletonOrThrow();
        // (batch, 1, atoms)
        var lab = massDist.singletonOrThrow();
        // (batch, 1, 1) - pred must be log softmax
        var loss = predLogits.logSoftmax(LAST_AXIS)
                .mul(lab)
                .neg()
                .sum(LAST_AXIS_ARR, KEEP_DIMS);
        // () -> scalar
        return loss.mean();
    }
}
