package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;

public interface IExperience extends AutoCloseable {

    NDArray state();

    ActionSpaceType.ActionResult action();

    double reward();

    NDArray nextState();

    boolean done();

    default <T> T actionAs(Class<T> clazz) {
        return action().valueAs(clazz);
    }

    @Override
    default void close() {
        action().close();
        state().close();
        nextState().close();
    }
}
