package br.com.guialves.rflr.gymnasium4j;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;

import java.awt.image.BufferedImage;
import java.util.Map;

public interface IEnv extends AutoCloseable {

    boolean closed();

    boolean scalarObservation();

    ActionSpaceType actionSpaceType();

    String actionSpaceStr();

    String observationSpaceStr();

    ActionSpaceType.ActionResult actionSpaceSample();

    Pair<Map<Object, Object>, NDArray> reset();

    EnvStepResult step(ActionSpaceType.ActionResult action) ;

    EnvStepResult step(ActionSpaceType.ActionResult action, NDManager manager);

    BufferedImage render();

    NDManager manager();
}
