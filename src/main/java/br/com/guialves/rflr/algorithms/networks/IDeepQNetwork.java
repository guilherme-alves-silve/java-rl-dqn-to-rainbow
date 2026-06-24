package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;

import java.nio.file.Path;

public interface IDeepQNetwork extends AutoCloseable {

    NDList forward(NDList input);

    NDArray forward(NDArray input);

    NDManager manager();

    void save(Path modelPath, String newModelName);

    IDeepQNetwork clone();

    Block getBlock();
}
