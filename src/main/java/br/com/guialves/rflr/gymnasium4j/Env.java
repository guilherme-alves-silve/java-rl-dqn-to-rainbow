package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.utils.ImageFromByteBuffer;
import br.com.guialves.rflr.gymnasium4j.utils.Numpy2DJLTypeMapper;
import br.com.guialves.rflr.python.PythonRuntime;
import lombok.Getter;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyObject;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonRuntime.eval;
import static br.com.guialves.rflr.python.numpy.NumpyByteBuffer.fillFromNumpy;
import static org.bytedeco.cpython.global.python.*;

@Slf4j
@Accessors(fluent = true)
public final class Env implements AutoCloseable {

    private final NDManager ndManager;
    private final PyObject env;

    private EnvStateMetadata stateMetadata;
    private EnvRenderMetadata renderMetadata;

    private ByteBuffer stateBuffer;
    private ByteBuffer imageBuffer;

    public Env(String envId, NDManager manager) {
        initPython();
        this.ndManager = manager.newSubManager();

        exec("""
        import gymnasium as gym
        env = gym.make('%s', render_mode='rgb_array')
        """.formatted(envId));

        env = eval("env");
    }

    public String actionSpaceStr() {
        try (var attr = PyObject_GetAttrString(env, "action_space");
             var out = PyObject_Str(attr)) {
            return str(out);
        }
    }

    public String observationSpaceStr() {
        try (var attr = PyObject_GetAttrString(env, "observation_space");
             var out = PyObject_Str(attr)) {
            return str(out);
        }
    }

    public int actionSpaceSample() {
        try (var sample = PyObject_CallMethod(env,
                "action_space.sample", null)) {
            return (int) toLong(sample);
        }
    }

    public Pair<Map<Object, Object>, NDArray> reset() {
        try (var pyReset = callMethod(env, "reset");
             var pyState = getItem(pyReset, 0);
             var pyInfo = getItem(pyReset, 1)) {

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

            var infoMap = pyDictToJava(pyInfo);
            return new Pair<>(infoMap, state);
        }
    }

    public EnvStepResult step(int action) {
        if (stateBuffer == null) {
            throw new IllegalStateException("You should call reset() first!");
        }

        System.out.println("EXEC 1");
        exec("with open('output_py.txt', 'w') as f:\n    f.write('starting...')");
        execIsolated("with open('output_py2.txt', 'w') as f:\n    f.write('starting...')");
        for (int i = 0; i < 1000; i++) {
            PythonRuntime.execIsolated("temp%d = ".formatted(i) + i);
        }
        //exec("""
        //with open('.output_py.txt', 'w') as f:
        //    try:
        //        f.write('starting...')
        //        #f.write(str(env))
        //        #f.write(str(env.step(1)))
        //    except Exception as ex:
        //        f.write(str(ex))
        //        f.write('end...')
        //""");
        System.out.println("EXEC 2");
        System.out.println(str(eval("env.step(1)")));
        /*
        try (var pyAction = pyLong(1);
             var pyStep = callMethod(env, "step", pyAction)) {
            var pyState = getItem(pyStep, 0);
            refInc(pyState);

            try (pyState;
                 var pyReward = getItem(pyStep, 1);
                 var pyTerminated = getItem(pyStep, 2);
                 var pyTruncated = getItem(pyStep, 3);
                 var pyInfoMap = getItem(pyStep, 4)) {

                double reward = toDouble(pyReward);
                boolean terminated = toBool(pyTerminated);
                boolean truncated = toBool(pyTruncated);
                var infoMap = pyDictToJava(pyInfoMap);

                fillFromNumpy(pyState, stateBuffer);

                var state = ndManager.create(
                        stateBuffer,
                        stateMetadata.djlShape,
                        stateMetadata.djlType
                );

                return new EnvStepResult(reward, terminated, truncated, infoMap)
                        .state(state);
            }
        }
         */
        return null;
    }

    public BufferedImage render() {
        try (var ndarray = callMethod(env, "render")) {

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
        try (var _ = callMethod(env, "close")) {
            log.info("Closed Env");
        } finally {
            ndManager.close();
            env.close();
            // TODO: Temp
            PythonRuntime.finalizePython();
        }
    }

    @Accessors(fluent = true)
    public static class EnvStateMetadata {

        protected final int[] shape;
        @Getter
        protected final String dtype;
        @Getter
        protected final Shape djlShape;
        @Getter
        protected final DataType djlType;
        @Getter
        protected final int size;

        static EnvStateMetadata fromNumpy(PyObject arr) {
            int[] shape = pyIntArrayToJava(attr(arr, "shape"));
            String dtype = attrStr(arr, "dtype");

            long[] longShape = Arrays.stream(shape).mapToLong(i -> i).toArray();
            DataType djlType = Numpy2DJLTypeMapper.numpyToDjl(dtype);

            int elements = Arrays.stream(shape).reduce(1, Math::multiplyExact);
            int size = elements * Numpy2DJLTypeMapper.bytesPerElement(dtype);

            return new EnvStateMetadata(shape, dtype,
                    new Shape(longShape), djlType, size);
        }

        private EnvStateMetadata(
                int[] shape,
                String dtype,
                Shape djlShape,
                DataType djlType,
                int size
        ) {
            this.shape = shape;
            this.dtype = dtype;
            this.djlShape = djlShape;
            this.djlType = djlType;
            this.size = size;
        }
    }

    @Accessors(fluent = true)
    public static class EnvRenderMetadata extends EnvStateMetadata {

        static EnvRenderMetadata fromNumpy(PyObject arr) {
            var base = EnvStateMetadata.fromNumpy(arr);
            return new EnvRenderMetadata(
                    base.shape,
                    base.dtype,
                    base.djlShape,
                    base.djlType,
                    base.size
            );
        }

        private EnvRenderMetadata(
                int[] shape,
                String dtype,
                Shape djlShape,
                DataType djlType,
                int size
        ) {
            super(shape, dtype, djlShape, djlType, size);
        }

        public int height() { return shape[0]; }
        public int width()  { return shape[1]; }
        public int channels() { return shape.length > 2 ? shape[2] : 1; }
    }
}
