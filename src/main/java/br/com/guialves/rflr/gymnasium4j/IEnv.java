package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;

import java.awt.image.BufferedImage;
import java.util.List;

public interface IEnv extends AutoCloseable {

    String envName();

    boolean closed();

    boolean scalarObservation();

    ActionSpaceType actionSpaceType();

    String actionSpaceStr();

    String observationSpaceStr();

    ActionSpaceType.ActionResult actionSpaceSample();

    EnvResetResult reset(NDManager manager);

    EnvStepResult step(ActionSpaceType.ActionResult action, NDManager sub);

    /**
     * Steps a vectorized environment, one action per environment.
     *
     * <p>The actions are given in the same order as the environments, so
     * {@code actions[i]} is the action for the i-th environment.</p>
     *
     * @param actions one action per environment
     * @param sub     the manager that owns the resulting observation batch
     * @return the batched step results
     */
    default VecEnvStepResults stepVec(ActionSpaceType.ActionResult actions, NDManager sub) {
        throw new IllegalArgumentException("This method stepVec, only works with vectorized environments!");
    }

    BufferedImage render();

    default List<BufferedImage> renderVec() {
        throw new IllegalArgumentException("This method renderVec, only works with vectorized environments!");
    }

    @Override
    void close();

    default void throwIfNotSupported(int actions) {
        var msg = "Incorrect number of actions for the environment %s: " +
                "Config=%s, Environment=%s";
        switch (actionSpaceType()) {
            case DISCRETE -> {
                var actionConfigSpaceStr = "Discrete(%s)".formatted(String.valueOf(actions));
                var actionSpaceStr = actionSpaceStr();
                if (!actionConfigSpaceStr.equals(actionSpaceStr)) {
                    throw new IllegalArgumentException(msg.formatted(envName(),
                            actionConfigSpaceStr, actionSpaceStr));
                }
            }
            case BOX, MULTI_DISCRETE, MULTI_BINARY, TEXT, UNKNOWN -> throw new UnsupportedOperationException("Not supported yet!");
        }
    }

    boolean vectorized();

    /**
     * The action space of a single environment.
     *
     * <p>For a vectorized environment the batched action space is a
     * {@code MultiDiscrete}, while the action space of one environment is still
     * a {@code Discrete}. This method exposes the latter, which is the one that
     * must be compared with the configuration.</p>
     *
     * @return the string representation of the action space of one environment
     */
    default String singleActionSpaceStr() {
        return actionSpaceStr();
    }
}
