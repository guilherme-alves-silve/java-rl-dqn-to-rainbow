package br.com.guialves.rflr.djlutils;

import ai.djl.nn.Block;
import ai.djl.training.optimizer.Optimizer;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;

@Slf4j
public class DJLOptimizer {

    private DJLOptimizer() {
        throw new IllegalStateException("No DJLOptimizer!");
    }

    public static boolean trainStep(Block block, Optimizer optimizer) {
        boolean atLeastOneGradient = false;
        for (var param : block.getParameters()) {
            if (!param.getValue().requiresGradient()) continue;
            atLeastOneGradient = true;
            var weight = param.getValue().getArray();
            @Cleanup var grad = weight.getGradient();
            optimizer.update(param.getKey(), weight, grad);
        }

        if (!atLeastOneGradient) log.warn("No training was made!");
        return atLeastOneGradient;
    }

    public static boolean trainStepClipGradients(Block block, Optimizer optimizer, float clipValue) {
        boolean atLeastOneGradient = false;
        for (var param : block.getParameters()) {
            if (!param.getValue().requiresGradient()) continue;
            atLeastOneGradient = true;
            var weight = param.getValue().getArray();
            try (var sub = subMgr(weight, "grad-clip")) {
                weight.tempAttach(sub);
                var grad = weight.getGradient();
                var norm = grad.norm();
                // getFloat()/toFloatArray() has a leak (GPU confirmed), solved with short temp sub manager
                float scalarNorm = norm.getFloat();
                if (scalarNorm > clipValue) {
                    grad.muli(clipValue / scalarNorm);
                }
                optimizer.update(param.getKey(), weight, grad);
            }
        }

        if (!atLeastOneGradient) log.warn("No training was made (grad clipping)!");
        return atLeastOneGradient;
    }
}
