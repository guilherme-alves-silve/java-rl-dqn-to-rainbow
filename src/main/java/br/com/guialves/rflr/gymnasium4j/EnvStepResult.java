package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;

import java.util.Map;

public record EnvStepResult(double reward,
                            boolean term,
                            boolean trunc,
                            Map<Object, Object> info,
                            NDArray state) implements AutoCloseable {

    public boolean done() {
        return term || trunc;
    }

    @Override
    public void close() {
        state.close();
    }
}
