package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import java.util.Map;

@Getter
@AllArgsConstructor
@Accessors(fluent = true)
@RequiredArgsConstructor
public class EnvStepResult {
    private final double reward;
    private final boolean term;
    private final boolean trunc;
    private final Map<Object, Object> info;
    // package private so only EnvProxy access this, but, be careful
    private NDArray state;

    EnvStepResult state(NDArray state) {
        this.state = state;
        return this;
    }

    public boolean done() {
        return term || trunc;
    }
}
