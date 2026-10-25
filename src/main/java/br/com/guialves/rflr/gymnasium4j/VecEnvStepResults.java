package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * I choose to convert the vectorized to a list of EnvStepResult to maintain
 * compatibility with the actual design.
 * @param steps
 */
public record VecEnvStepResults(List<EnvStepResult> steps,
                                Map<Object, Object> info) implements AutoCloseable {

    VecEnvStepResults(double[] rewards,
                      boolean[] terms,
                      boolean[] truncs,
                      Map<Object, Object> info,
                      NDArray states) {
        this(new ArrayList<>(rewards.length), info);
        for (int i = 0; i < rewards.length; ++i) {
            var step = new EnvStepResult(
                    rewards[i],
                    terms[i],
                    truncs[i],
                    info,
                    states.get(i)
            );
            steps.add(step);
        }
    }

    public boolean anyTerm() {
        for (var step : steps) {
            if (step.term()) return true;
        }

        return false;
    }

    public boolean anyTrunc() {
        for (var step : steps) {
            if (step.trunc()) return true;
        }

        return false;
    }

    public boolean anyDone() {
        for (var step : steps) {
            if (step.done()) return true;
        }

        return false;
    }

    @Override
    public void close() {
        for (var step : steps) {
            step.close();
        }
    }
}
