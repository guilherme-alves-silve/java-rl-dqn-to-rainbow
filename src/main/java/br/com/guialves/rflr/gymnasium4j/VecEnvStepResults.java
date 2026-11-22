package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * I choose to convert the vectorized to a list of EnvStepResult to maintain
 * compatibility with the actual design.
 *
 * @param steps one step result per environment
 * @param info  the batched info dictionary returned by the vector environment
 * @param states the batched observation, shape {@code (numEnvs, ...)}
 */
public record VecEnvStepResults(List<EnvStepResult> steps,
                                Map<Object, Object> info,
                                NDArray states) implements AutoCloseable {

    VecEnvStepResults(double[] rewards,
                      boolean[] terms,
                      boolean[] truncs,
                      Map<Object, Object> info,
                      NDArray states) {
        this(buildSteps(rewards, terms, truncs, info, states), info, states);
    }

    private static List<EnvStepResult> buildSteps(double[] rewards,
                                                  boolean[] terms,
                                                  boolean[] truncs,
                                                  Map<Object, Object> info,
                                                  NDArray states) {
        var steps = new ArrayList<EnvStepResult>(rewards.length);
        for (int i = 0; i < rewards.length; ++i) {
            steps.add(new EnvStepResult(
                    rewards[i],
                    terms[i],
                    truncs[i],
                    info,
                    states.get(i)
            ));
        }
        return steps;
    }

    public EnvStepResult step(int idx) {
        return steps.get(idx);
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
