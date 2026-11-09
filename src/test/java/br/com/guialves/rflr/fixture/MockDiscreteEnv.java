package br.com.guialves.rflr.fixture;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import lombok.Cleanup;

import java.awt.image.BufferedImage;
import java.util.Collections;

/**
 * Minimal {@link IEnv} that just keeps cycling for the duration of a single
 * {@code selectAction} call. The agent only ever invokes
 * {@code actionSpaceType()} on the env during selection, so this is enough to
 * satisfy the constructor.
 */
public final class MockDiscreteEnv implements IEnv {

    private final int obsDim;
    private final int nActions;
    private final ActionSpaceType type = ActionSpaceType.DISCRETE;

    public MockDiscreteEnv(int obsDim, int nActions) {
        this.obsDim = obsDim;
        this.nActions = nActions;
    }

    @Override
    public String envName() {
        return MockDiscreteEnv.class.getName();
    }

    @Override
    public boolean closed() {
        return false;
    }

    @Override
    public boolean scalarObservation() {
        return true;
    }

    @Override
    public ActionSpaceType actionSpaceType() {
        return type;
    }

    @Override
    public String actionSpaceStr() {
        return "Discrete(" + nActions + ")";
    }

    @Override
    public String observationSpaceStr() {
        return "Box(" + obsDim + ")";
    }

    @Override
    public ActionSpaceType.ActionResult actionSpaceSample() {
        return type.get(0);
    }

    @Override
    public EnvResetResult reset(NDManager manager) {
        var state = manager.zeros(new Shape(obsDim));
        return new EnvResetResult(state,
                Collections.emptyMap());
    }

    @Override
    public EnvStepResult step(ActionSpaceType.ActionResult action, NDManager sub) {
        @Cleanup var next = sub.zeros(new Shape(obsDim));
        return new EnvStepResult(0.0, false, false, Collections.emptyMap(), next);
    }

    @Override
    public BufferedImage render() {
        return null;
    }

    @Override
    public void close() {
        // Ignored
    }
}
