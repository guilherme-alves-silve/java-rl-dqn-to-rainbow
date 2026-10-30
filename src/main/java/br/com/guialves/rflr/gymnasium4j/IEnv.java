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

    default VecEnvStepResults stepVec(ActionSpaceType.ActionResult action, NDManager sub) {
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
}
