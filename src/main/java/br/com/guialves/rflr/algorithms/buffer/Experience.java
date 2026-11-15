package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;

public record Experience(int envNumber,
                         NDArray state,
                         ActionResult action,
                         double reward,
                         NDArray nextState,
                         boolean done) implements AutoCloseable {

    public static final int DEFAULT_ENV_NUMBER = 0;

    public Experience(NDArray state, ActionResult action, double reward, NDArray nextState, boolean done) {
        this(DEFAULT_ENV_NUMBER, state, action, reward, nextState, done);
    }

    public <T> T actionAs(Class<T> clazz) {
        return action.valueAs(clazz);
    }

    @Override
    public void close() {
        state.close();
        action.close();
        nextState.close();
    }
}
