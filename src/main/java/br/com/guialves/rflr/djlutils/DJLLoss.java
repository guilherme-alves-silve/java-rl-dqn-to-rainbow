package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.loss.Loss;
import lombok.Cleanup;

import java.util.function.Function;
import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static br.com.guialves.rflr.gymnasium4j.EngineUtils.gradient;

/**
 * Utility for loss computation with automatic memory cleanup.
 * All intermediate NDArrays are closed after use via @Cleanup and sub-managers.
 *
 * <p><b>Memory Management:</b>
 * <ul>
 *   <li>Creates a sub-manager for temporary allocations</li>
 *   <li>Uses @Cleanup to close intermediate NDArrays</li>
 *   <li>Returns a primitive float (not an NDArray)</li>
 * </ul>
 */
public class DJLLoss {

    private DJLLoss() {
        throw new IllegalStateException("No DJLLoss!");
    }

    /**
     * Computes loss and backpropagation, returning the mean loss as a float.
     *
     * <p>This method ensures that all objects allocated for gradient calculation
     * are properly deallocated after each iteration, preventing memory accumulation
     * during training loops.
     *
     * <p><b>Why not use EasyTrain?</b>
     * EasyTrain does not meet the requirements of this project (custom PER loss,
     * priority updates, and fine-grained control over the training loop).
     *
     * @param manager    parent NDManager
     * @param lossFunc   loss function (e.g., MSE, PERL2Loss)
     * @param yTarget    target values (labels)
     * @param yPredBlock forward pass function that returns predictions
     * @return mean loss value as float
     */
    public static float backwardLoss(final NDManager manager,
                                     final Loss lossFunc,
                                     final NDArray yTarget,
                                     final Supplier<NDArray> yPredBlock) {
        @Cleanup var sub = subMgr(manager, "scoped-back");
        @Cleanup var gradCol = gradient();
        var yPred = yPredBlock.get();
        var lossesVal =  evaluate(lossFunc, sub, gradCol, yTarget, yPred);
        return lossesVal.stopGradient().mean().getFloat();
    }

    /**
     * Computes loss and backpropagation, returning the mean loss as a float.
     *
     * <p>This method ensures that all objects allocated for gradient calculation
     * are properly deallocated after each iteration, preventing memory accumulation
     * during training loops.
     *
     * <p><b>Why not use EasyTrain?</b>
     * EasyTrain does not meet the requirements of this project (custom PER loss,
     * priority updates, and fine-grained control over the training loop).
     *
     * @param manager    parent NDManager
     * @param lossFunc   loss function (e.g., MSE, PERL2Loss)
     * @param yTarget    target values (labels)
     * @param yPredBlock forward pass function that returns predictions
     * @param arrays     additional NDArrays needed by blockYPred (states, actions, etc.)
     * @return mean loss value as float
     */
    public static float backwardLoss(final NDManager manager,
                                     final Loss lossFunc,
                                     final NDArray yTarget,
                                     final Function<NDArray[], NDArray> yPredBlock,
                                     final NDArray... arrays) {
        @Cleanup var sub = subMgr(manager, "scoped-back");
        @Cleanup var gradCol = gradient();
        sub.tempAttachAll(arrays);
        var lossesVal = evaluate(lossFunc, sub, gradCol, yTarget, yPredBlock, arrays);
        return lossesVal.stopGradient().mean().getFloat();
    }

    /**
     * Computes loss and backpropagation, returning raw per-sample losses.
     *
     * <p>This method is specifically designed for custom loss functions.
     *
     * @param manager    parent NDManager
     * @param lossFunc   loss function (must support per-sample weights, e.g., PERL2Loss)
     * @param yTarget    target values (labels)
     * @param yPredBlock forward pass function that returns predictions
     * @param arrays     additional NDArrays needed by blockYPred (states, actions, etc.)
     * @return per-sample losses with shape (batchSize, 1), gradients stopped
     */
    public static NDArray rawBackwardLoss(final NDManager manager,
                                          final Loss lossFunc,
                                          final NDArray yTarget,
                                          final Function<NDArray[], NDArray> yPredBlock,
                                          final NDArray... arrays) {
        @Cleanup var sub = subMgr(manager, "scoped-back");
        @Cleanup var gradCol = gradient();
        sub.tempAttachAll(arrays);
        var lossesVal = evaluate(lossFunc, sub, gradCol, yTarget, yPredBlock, arrays);
        return manager.ret(lossesVal.stopGradient());
    }

    public static NDArray rawBackwardLoss(final NDManager manager,
                                          final Loss lossFunc,
                                          final NDArray yTarget,
                                          final Supplier<NDArray> yPredBlock) {
        @Cleanup var sub = subMgr(manager, "scoped-back");
        @Cleanup var gradCol = gradient();
        var yPred = yPredBlock.get();
        var lossesVal = evaluate(lossFunc, sub, gradCol, yTarget, yPred);
        return manager.ret(lossesVal.stopGradient());
    }

    /**
     * Internal method that evaluates the loss and performs backpropagation.
     *
     * <p>Temporarily attaches all arrays to the sub-manager, computes the loss,
     * and executes backward pass to compute gradients.
     *
     * @param lossFunc   loss function to evaluate
     * @param sub        sub-manager for temporary memory management
     * @param gradCol    gradient collector for backpropagation
     * @param yTarget    target values
     * @param yPredBlock predictions from the forward pass
     * @param arrays     additional arrays used in the computation
     * @return raw loss NDArray (still managed by the sub-manager)
     */
    private static NDArray evaluate(final Loss lossFunc,
                                    final NDManager sub,
                                    final GradientCollector gradCol,
                                    final NDArray yTarget,
                                    final Function<NDArray[], NDArray> yPredBlock,
                                    final NDArray[] arrays) {
        var yPred = yPredBlock.apply(arrays);
        return evaluate(lossFunc, sub, gradCol, yTarget, yPred);
    }

    private static NDArray evaluate(final Loss lossFunc,
                                    final NDManager sub,
                                    final GradientCollector gradCol,
                                    final NDArray yTarget,
                                    final NDArray yPred) {
        yTarget.tempAttach(sub);
        yPred.attach(sub);
        var lossesVal = lossFunc.evaluate(new NDList(yTarget), new NDList(yPred));
        gradCol.backward(lossesVal);
        return lossesVal;
    }
}
