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
import lombok.Cleanup;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.initGradients;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DJLOptimizerTest {

    private static final float TEST_CLIP_GRAD_THRESHOLD = 1f;

    @Test
    void shouldNotKeepGrowingAfterWarmupWhenUsingBackward() {
        @Cleanup var manager = NDManager.newBaseManager();
        var block = new SequentialBlock()
                .add(Linear.builder().setUnits(8).optBias(true).build())
                .add(Linear.builder().setUnits(2).optBias(true).build());

        block.initialize(manager, DataType.FLOAT32, new Shape(1, 4));
        initGradients(block);
        var parameterStore = new ParameterStore(manager, false);
        var optimizer = Adam.builder()
                .optLearningRateTracker(Tracker.fixed(0.01f))
                .build();

        var loss = Loss.l2Loss();
        var x = setName(manager.ones(new Shape(1, 4)), "x");
        var y = setName(manager.zeros(new Shape(1, 2)), "y");

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

    @Test
    void shouldNotKeepGrowingAfterWarmupWhenUsingBackwardClipGrad() {
        @Cleanup var manager = NDManager.newBaseManager();
        var block = new SequentialBlock()
                .add(Linear.builder().setUnits(8).optBias(true).build())
                .add(Linear.builder().setUnits(2).optBias(true).build());

        block.initialize(manager, DataType.FLOAT32, new Shape(1, 4));
        initGradients(block);

        var parameterStore = new ParameterStore(manager, false);
        var optimizer = Adam.builder()
                .optLearningRateTracker(Tracker.fixed(0.01f))
                .build();

        var loss = Loss.l2Loss();
        var x = setName(manager.ones(new Shape(1, 4)), "x");
        var y = setName(manager.zeros(new Shape(1, 2)), "y");

        runBackwardLossAndStepClipGrad(manager, block, parameterStore, optimizer, loss, x, y);
        runBackwardLossAndStepClipGrad(manager, block, parameterStore, optimizer, loss, x, y);

        int afterWarmup = managedArrayCount(manager);

        runBackwardLossAndStepClipGrad(manager, block, parameterStore, optimizer, loss, x, y);
        int afterStep1 = managedArrayCount(manager);

        runBackwardLossAndStepClipGrad(manager, block, parameterStore, optimizer, loss, x, y);
        int afterStep2 = managedArrayCount(manager);

        runBackwardLossAndStepClipGrad(manager, block, parameterStore, optimizer, loss, x, y);
        int afterStep3 = managedArrayCount(manager);

        assertTrue(afterWarmup > 0);
        debugDump(manager);
        assertEquals(afterWarmup, afterStep1);
        assertEquals(afterStep1, afterStep2);
        assertEquals(afterStep2, afterStep3);
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
            var out = block.forward(parameterStore, new NDList(x), true).singletonOrThrow();
            return setName(out, "out-1");
        }, x);

        backwardLoss(manager, loss, y, arrays -> {
            var gradX = arrays[0];
            var out = block.forward(parameterStore, new NDList(gradX), true).singletonOrThrow();
            return setName(out, "out-2");
        }, x);

        var trained = DJLOptimizer.trainStep(block, optimizer);
        assertTrue(trained);
    }

    private void runBackwardLossAndStepClipGrad(NDManager manager,
                                                SequentialBlock block,
                                                ParameterStore parameterStore,
                                                Optimizer optimizer,
                                                Loss loss,
                                                NDArray x,
                                                NDArray y) {
        backwardLoss(manager, loss, y, arrays -> {
            var gradX = arrays[0];
            IO.println("gradX: " + gradX);
            var out = block.forward(parameterStore, new NDList(x), true).singletonOrThrow();
            return setName(out, "out-clip-1");
        }, x);

        backwardLoss(manager, loss, y, arrays -> {
            var gradX = arrays[0];
            var out = block.forward(parameterStore, new NDList(gradX), true).singletonOrThrow();
            return setName(out, "out-clip-2");
        }, x);

        var trained = DJLOptimizer.trainStepClipGradients(block, optimizer, TEST_CLIP_GRAD_THRESHOLD);
        assertTrue(trained);
    }
}
