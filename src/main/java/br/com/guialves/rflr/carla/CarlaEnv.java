package br.com.guialves.rflr.carla;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;

import java.awt.image.BufferedImage;

public class CarlaEnv implements IEnv {

    @Override
    public boolean closed() {
        return false;
    }

    @Override
    public boolean scalarObservation() {
        return false;
    }

    @Override
    public ActionSpaceType actionSpaceType() {
        return null;
    }

    @Override
    public String actionSpaceStr() {
        return "";
    }

    @Override
    public String observationSpaceStr() {
        return "";
    }

    @Override
    public ActionSpaceType.ActionResult actionSpaceSample() {
        return null;
    }

    @Override
    public EnvResetResult reset(NDManager manager) {
        return null;
    }

    @Override
    public EnvStepResult step(ActionSpaceType.ActionResult action, NDManager sub) {
        return null;
    }

    @Override
    public BufferedImage render() {
        return null;
    }

    @Override
    public void close() {

    }
}
