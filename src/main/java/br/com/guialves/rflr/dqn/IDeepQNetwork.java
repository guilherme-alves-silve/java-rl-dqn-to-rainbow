package br.com.guialves.rflr.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.SneakyThrows;

import java.nio.file.Path;

public interface IDeepQNetwork extends AutoCloseable {

    NDList forward(NDList input);

    NDArray forward(NDArray input);

    NDManager manager();

    @SneakyThrows
    void save(Path modelPath, String newModelName);

    IDeepQNetwork clone();
}
