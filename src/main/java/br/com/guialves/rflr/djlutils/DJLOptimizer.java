package br.com.guialves.rflr.djlutils;

import ai.djl.nn.Block;
import ai.djl.training.optimizer.Optimizer;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

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
}
