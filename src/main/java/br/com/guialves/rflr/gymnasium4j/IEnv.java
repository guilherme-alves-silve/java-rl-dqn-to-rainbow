package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;
import ai.djl.util.Pair;

import java.awt.image.BufferedImage;
import java.util.Map;

public interface IEnv extends AutoCloseable {

    boolean closed();

    boolean discreteObservation();

    String actionSpaceStr();

    String observationSpaceStr();

    ActionSpaceType.ActionResult actionSpaceSample();

    Pair<Map<Object, Object>, NDArray> reset();

    EnvStepResult step(ActionSpaceType.ActionResult action) ;

    BufferedImage render();
}
