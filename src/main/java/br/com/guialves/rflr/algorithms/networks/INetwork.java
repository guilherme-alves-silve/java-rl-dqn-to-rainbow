package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Cleanup;

import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;

public interface INetwork extends AutoCloseable {

    NDList forward(NDList input);

    NDArray forward(NDArray input);

    NDManager subManager();

    default NDArray forward(NDArray input, final UnaryOperator<NDArray> block) {
        return scoped(it -> {
            @Cleanup var out = forward(it);
            out.tempAttach(it.getManager());
            return block.apply(out);
        }, input);
    }

    default NDArray forward(NDArray input, final BiFunction<NDArray, NDArray[], NDArray> block, final NDArray... arrays) {
        return scoped((itInput, itArrays) -> {
            @Cleanup var out = forward(itInput);
            out.tempAttach(itInput.getManager());
            return block.apply(out, itArrays);
        }, input, arrays);
    }
}
