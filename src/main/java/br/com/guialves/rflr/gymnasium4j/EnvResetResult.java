package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;

import java.util.Map;

public record EnvResetResult(NDArray state,
                             Map<Object, Object> info) implements AutoCloseable {
    @Override
    public void close() {
        state.close();
    }
}
