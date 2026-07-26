package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;

public record Experience(NDArray state,
                         ActionResult action,
                         double reward,
                         NDArray nextState,
                         boolean done) implements AutoCloseable {

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
