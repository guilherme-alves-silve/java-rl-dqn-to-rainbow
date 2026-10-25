package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;

import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.List;

public interface IEnv extends AutoCloseable {

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
}
