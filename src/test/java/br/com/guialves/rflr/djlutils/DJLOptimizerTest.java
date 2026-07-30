package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.setGradients;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DJLOptimizerTest {

    @Test
    void shouldNotKeepGrowingAfterWarmupWhenUsingBackward() {
        try (var manager = NDManager.newBaseManager()) {
            var block = new SequentialBlock()
                    .add(Linear.builder().setUnits(8).optBias(true).build())
                    .add(Linear.builder().setUnits(2).optBias(true).build());

            block.initialize(manager, DataType.FLOAT32, new Shape(1, 4));
            setGradients(block);

            var parameterStore = new ParameterStore(manager, false);
            var optimizer = Adam.builder()
                    .optLearningRateTracker(Tracker.fixed(0.01f))
                    .build();

            var loss = Loss.l2Loss();
            var x = manager.ones(new Shape(1, 4));
            var y = manager.zeros(new Shape(1, 2));

            runBackwardLossAndStep(manager, block, parameterStore, optimizer, loss, x, y);
            runBackwardLossAndStep(manager, block, parameterStore, optimizer, loss, x, y);

            int afterWarmup = managedArrayCount(manager);

            runBackwardLossAndStep(manager, block, parameterStore, optimizer, loss, x, y);
            int afterStep1 = managedArrayCount(manager);

            runBackwardLossAndStep(manager, block, parameterStore, optimizer, loss, x, y);
            int afterStep2 = managedArrayCount(manager);

            runBackwardLossAndStep(manager, block, parameterStore, optimizer, loss, x, y);
            int afterStep3 = managedArrayCount(manager);

            assertTrue(afterWarmup > 0);
            debugDump(manager);
            assertEquals(afterWarmup, afterStep1);
            assertEquals(afterStep1, afterStep2);
            assertEquals(afterStep2, afterStep3);
        }
    }

    private void runBackwardLossAndStep(NDManager manager,
                                        SequentialBlock block,
                                        ParameterStore parameterStore,
                                        Optimizer optimizer,
                                        Loss loss,
                                        NDArray x,
                                        NDArray y) {
        backwardLoss(manager, loss, y, arrays -> {
            var gradX = arrays[0];
            IO.println("gradX: " + gradX);
            return block.forward(parameterStore, new NDList(x), true).singletonOrThrow();
        }, x);

        backwardLoss(manager, loss, y, arrays -> {
            var gradX = arrays[0];
            return block.forward(parameterStore, new NDList(gradX), true).singletonOrThrow();
        }, x);

        DJLOptimizer.trainStep(block, optimizer);
    }
}
