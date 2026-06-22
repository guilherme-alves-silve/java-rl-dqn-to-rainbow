package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;

import java.awt.image.BufferedImage;
import java.util.Collections;

public interface IEnv extends AutoCloseable {

    boolean closed();

    boolean scalarObservation();

    ActionSpaceType actionSpaceType();

    String actionSpaceStr();

    String observationSpaceStr();

    ActionSpaceType.ActionResult actionSpaceSample();

    EnvResetResult reset();

    EnvStepResult step(ActionSpaceType.ActionResult action);

    EnvStepResult step(ActionSpaceType.ActionResult action, NDManager manager);

    BufferedImage render();

    NDManager manager();

    @Override
    void close();
}
