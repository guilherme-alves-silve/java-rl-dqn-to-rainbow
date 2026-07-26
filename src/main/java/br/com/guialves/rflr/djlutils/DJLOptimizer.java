package br.com.guialves.rflr.djlutils;

import ai.djl.nn.Block;
import ai.djl.training.optimizer.Optimizer;
import lombok.Cleanup;

public class DJLOptimizer {

    private DJLOptimizer() {
        throw new IllegalStateException("No DJLOptimizer!");
    }

    public static void trainStep(Block block, Optimizer optimizer) {
        for (var param : block.getParameters()) {
            if (!param.getValue().requiresGradient()) continue;
            var weight = param.getValue().getArray();
            @Cleanup var grad = weight.getGradient();
            optimizer.update(param.getKey(), weight, grad);
        }
    }
}
