package br.com.guialves.rflr.gymnasium4j;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.utils.ImageFromByteBuffer;
import lombok.Getter;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyObject;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;
import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.detectActionSpaceType;
import static br.com.guialves.rflr.python.PythonDataStructures.*;
import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.numpy.NumPyByteBuffer.fillFromNumpy;

@Slf4j
@Accessors(fluent = true)
public final class Env implements IEnv {

    private static final boolean DEBUG = true;
    private final NDManager manager;
    @Getter
    private final String varEnvCode;
    @Getter
    private final String envName;
    private final PyObject pyEnv;
    private final PyObject pyActionSpace;
    private final PyObject pyObservationSpace;
    private final PyObject pyRender;
    private final PyObject pyStep;
    private final PyObject pyReset;
    private final ActionSpaceType actionSpaceType;

    @Getter
    private boolean closed;
    private EnvStateMetadata stateMetadata;
    private EnvRenderMetadata renderMetadata;

    private ByteBuffer stateBuffer;
    private ByteBuffer imageBuffer;
    private boolean discreteObservation;

    Env(String varEnvCode, String envName, String generatedScript, NDManager manager) {
        initPython();
        this.varEnvCode = varEnvCode;
        this.envName = envName;
        this.manager = manager.newSubManager();
        exec(generatedScript);

        this.pyEnv = eval("env_" + varEnvCode);
        this.pyActionSpace = attr(pyEnv, "action_space");
        this.actionSpaceType = detectActionSpaceType(pyActionSpace);
        this.pyObservationSpace = attr(pyEnv, "observation_space");
        this.pyRender = attr(pyEnv, "render");
        this.pyStep = attr(pyEnv, "step");
        this.pyReset = attr(pyEnv, "reset");
    }

    @Override
    public boolean discreteObservation() {
        return discreteObservation;
    }

    @Override
    public String actionSpaceStr() {
        return toStr(pyActionSpace);
    }

    @Override
    public String observationSpaceStr() {
        return toStr(pyObservationSpace);
    }

    @Override
    public ActionResult actionSpaceSample() {
        try (var pySample = callMethod(pyActionSpace, "sample")) {
            return actionSpaceType.convert(pySample);
        }
    }

    @Override
    public Pair<Map<Object, Object>, NDArray> reset() {
        try (var result = callFunction(pyReset)) {

            var pyState = getItem(result, 0);
            var infoMap = getItemMap(result, 1);

            if (!hasAttr(pyState, "shape")) {
                this.discreteObservation = true;
                long observationValue = toLong(pyState);
                var state = manager.create(observationValue);
                log.debug("Discrete observation: {}", observationValue);
                return new Pair<>(infoMap, state);
            } else {
                discreteObservation = false;
                if (stateMetadata == null) {
                    stateMetadata = EnvStateMetadata.fromNumpy(pyState);
                    stateBuffer = ByteBuffer
                            .allocate(stateMetadata.size())
                            .order(ByteOrder.nativeOrder());
                }

                fillFromNumpy(pyState, stateBuffer);

                var state = manager.create(
                        stateBuffer,
                        stateMetadata.djlShape(),
                        stateMetadata.djlType
                );

                return new Pair<>(infoMap, state);
            }
        }
    }

    @Override
    public EnvStepResult step(ActionResult action) {
        return step(action, manager);
    }

    @Override
    public EnvStepResult step(ActionResult action, NDManager manager) {
        try (var result = callFunction(pyStep, action.pyObj)) {
            NDArray state;

            if (discreteObservation) {
                var pyState = getItem(result, 0);
                long observationValue = toLong(pyState);
                state = manager.create(observationValue);

                log.debug("Discrete observation after step: {}", observationValue);
            } else {
                if (stateBuffer == null) {
                    throw new IllegalStateException("You should call reset() first!");
                }

                fillFromNumpy(getItem(result, 0), stateBuffer);
                state = manager.create(
                        stateBuffer,
                        stateMetadata.djlShape,
                        stateMetadata.djlType
                );
            }

            double reward = getItemDouble(result, 1);
            boolean terminated = getItemBool(result, 2);
            boolean truncated = getItemBool(result, 3);
            var infoMap = getItemMap(result, 4);

            return new EnvStepResult(reward, terminated, truncated, infoMap)
                    .state(state);
        }
    }

    @Override
    public BufferedImage render() {
        try (var ndarray = callFunction(pyRender)) {
            if (renderMetadata == null) {
                renderMetadata = EnvRenderMetadata.fromNumpy(ndarray);
                imageBuffer = ByteBuffer
                        .allocateDirect(renderMetadata.size())
                        .order(ByteOrder.nativeOrder());
            }

            fillFromNumpy(ndarray, imageBuffer);

            return ImageFromByteBuffer.byteBufferToImage(
                    imageBuffer,
                    renderMetadata.width(),
                    renderMetadata.height(),
                    renderMetadata.channels() == 4
            );
        }
    }

    @Override
    public NDManager manager() {
        return manager;
    }

    @Override
    public void close() {
        if (closed) {
            log.warn("The env_{} was already closed!", varEnvCode);
            return;
        }
        this.closed = true;

        if (DEBUG) log.info("Before close - pyEnv: {}, pyActionSpace: {}, pyObservationSpace: {}, pyRender: {}, pyStep: {}, pyReset: {}",
                refCount(pyEnv), refCount(pyActionSpace), refCount(pyObservationSpace),
                refCount(pyRender), refCount(pyStep), refCount(pyReset));

        decRef(pyActionSpace);
        decRef(pyObservationSpace);
        decRef(pyRender);
        decRef(pyStep);
        decRef(pyReset);
        decRef(pyEnv);

        exec("if 'env' in globals(): del env");

        if (DEBUG) log.info("After close - pyEnv: {}, pyActionSpace: {}, pyObservationSpace: {}, pyRender: {}, pyStep: {}, pyReset: {}",
                    refCount(pyEnv), refCount(pyActionSpace), refCount(pyObservationSpace),
                    refCount(pyRender), refCount(pyStep), refCount(pyReset));

        log.info("Closed Env");
        manager.close();
    }
}
