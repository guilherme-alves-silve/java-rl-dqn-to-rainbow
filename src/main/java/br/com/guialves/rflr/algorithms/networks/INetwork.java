package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.function.UnaryOperator;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;

public interface INetwork extends AutoCloseable {

    NDList forward(NDList input);

    NDArray forward(NDArray input);

    NDManager manager();

    default NDArray forward(NDArray input, final UnaryOperator<NDArray> block) {
        return scoped(it -> block.apply(forward(it)), input);
    }
}
