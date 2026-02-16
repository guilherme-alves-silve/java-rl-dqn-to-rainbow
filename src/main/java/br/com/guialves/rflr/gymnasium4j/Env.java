package br.com.guialves.rflr.gymnasium4j;

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
import java.util.UUID;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.*;
import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonRuntime.eval;
import static br.com.guialves.rflr.python.numpy.NumpyByteBuffer.fillFromNumpy;

@Slf4j
@Accessors(fluent = true)
public final class Env implements AutoCloseable {

    private static final boolean DEBUG = true;
    private final NDManager ndManager;
    private final String varEnvCode;
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

    public Env(String envId, NDManager manager) {
        initPython();
        this.ndManager = manager.newSubManager();
        this.varEnvCode = UUID.randomUUID().toString().replace("-", "");
        exec("""
        import gymnasium as gym
        env_%s = gym.make('%s', render_mode='rgb_array')
        """.formatted(varEnvCode, envId));

        this.pyEnv = eval("env_" + varEnvCode);
        this.pyActionSpace = attr(pyEnv, "action_space");
        this.actionSpaceType = detectActionSpaceType(pyActionSpace);
        this.pyObservationSpace = attr(pyEnv, "observation_space");
        this.pyRender = attr(pyEnv, "render");
        this.pyStep = attr(pyEnv, "step");
        this.pyReset = attr(pyEnv, "reset");
    }

    public String actionSpaceStr() {
        return str(pyActionSpace);
    }

    public String observationSpaceStr() {
        return str(pyObservationSpace);
    }

    public ActionResult actionSpaceSample() {
        try (var pySample = callMethod(pyActionSpace, "sample")) {
            return actionSpaceType.convert(pySample);
        }
    }

    public Pair<Map<Object, Object>, NDArray> reset() {
        try (var result = callFunction(pyReset)) {

            var pyState = getItem(result, 0);
            var infoMap = getItemMap(result, 1);

            if (stateMetadata == null) {
                stateMetadata = EnvStateMetadata.fromNumpy(pyState);
                stateBuffer = ByteBuffer
                        .allocate(stateMetadata.size())
                        .order(ByteOrder.nativeOrder());
            }

            fillFromNumpy(pyState, stateBuffer);

            var state = ndManager.create(
                    stateBuffer,
                    stateMetadata.djlShape(),
                    stateMetadata.djlType
            );

            return new Pair<>(infoMap, state);
        }
    }

    public EnvStepResult step(ActionResult action) {
        if (stateBuffer == null) {
            throw new IllegalStateException("You should call reset() first!");
        }

        try (var result = callFunction(pyStep, action.pyObj)) {
            fillFromNumpy(getItem(result, 0), stateBuffer);
            double reward = getItemDouble(result, 1);
            boolean terminated = getItemBool(result, 2);
            boolean truncated = getItemBool(result, 3);
            var infoMap = getItemMap(result, 4);

            var state = ndManager.create(
                    stateBuffer,
                    stateMetadata.djlShape,
                    stateMetadata.djlType
            );

            return new EnvStepResult(reward, terminated, truncated, infoMap)
                    .state(state);
        }
    }

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
    public void close() {
        if (closed) {
            log.warn("The env_{} was already closed!", varEnvCode);
            return;
        }
        this.closed = true;

        if (DEBUG) log.info("Before close - pyEnv: {}, pyActionSpace: {}, pyObservationSpace: {}, pyRender: {}, pyStep: {}, pyReset: {}",
                refCount(pyEnv), refCount(pyActionSpace), refCount(pyObservationSpace),
                refCount(pyRender), refCount(pyStep), refCount(pyReset));

        refDec(pyActionSpace);
        refDec(pyObservationSpace);
        refDec(pyRender);
        refDec(pyStep);
        refDec(pyReset);
        refDec(pyEnv);

        exec("if 'env' in globals(): del env");

        if (DEBUG) log.info("After close - pyEnv: {}, pyActionSpace: {}, pyObservationSpace: {}, pyRender: {}, pyStep: {}, pyReset: {}",
                    refCount(pyEnv), refCount(pyActionSpace), refCount(pyObservationSpace),
                    refCount(pyRender), refCount(pyStep), refCount(pyReset));

        log.info("Closed Env");
        ndManager.close();
    }
}
