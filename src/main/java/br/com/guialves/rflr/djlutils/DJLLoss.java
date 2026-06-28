package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import lombok.Cleanup;

import java.util.function.Function;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scopedToFloat;
import static br.com.guialves.rflr.gymnasium4j.EngineUtils.gradient;

public class DJLLoss {

    private DJLLoss() {
        throw new IllegalStateException("No DJLLoss!");
    }

    public static float backwardLoss(NDManager manager,
                                     Loss lossFunc,
                                     NDArray y,
                                     final Function<NDArray[], NDArray> blockPred,
                                     final NDArray... arrays) {
        try (var sub = manager.newSubManager();
             var grad = gradient()) {

            y.tempAttach(sub);
            sub.tempAttachAll(arrays);

            @Cleanup var yPred = blockPred.apply(arrays);
            yPred.tempAttach(sub);

            @Cleanup var lossVal = lossFunc.evaluate(new NDList(y), new NDList(yPred));

            grad.backward(lossVal);
            return scopedToFloat(it -> it.stopGradient().mean(), lossVal);
        }
    }
}
