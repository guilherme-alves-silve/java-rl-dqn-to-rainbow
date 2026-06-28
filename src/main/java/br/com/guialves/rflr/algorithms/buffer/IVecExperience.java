package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;

public interface IVecExperience extends AutoCloseable {

    NDArray states();

    NDArray actions();

    NDArray rewards();

    NDArray nextStates();

    NDArray dones();

    @Override
    default void close() {
        states().close();
        actions().close();
        rewards().close();
        nextStates().close();
        dones().close();
    }
}
