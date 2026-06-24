package br.com.guialves.rflr.gymnasium4j;

import ai.djl.nn.Block;
import ai.djl.training.optimizer.Optimizer;

public class OptimizerUtils {

    private OptimizerUtils() {
        throw new IllegalArgumentException("No OptimizerUtils!");
    }

    public static void trainStep(Block block, Optimizer optimizer) {
        for (var param : block.getParameters()) {
            if (!param.getValue().requiresGradient()) continue;
            var weight = param.getValue().getArray();
            var grad = weight.getGradient();
            optimizer.update(param.getKey(), weight, grad);
        }
    }
}
