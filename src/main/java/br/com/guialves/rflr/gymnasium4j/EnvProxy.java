package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.utils.GymPythonLauncher;
import br.com.guialves.rflr.gymnasium4j.utils.ImageFromByteBuffer;
import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import br.com.guialves.rflr.gymnasium4j.utils.Numpy2DJLTypeMapper;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.gson.Gson;
import io.vavr.control.Try;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.zeromq.ZContext;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;

import static br.com.guialves.rflr.gymnasium4j.transform.EnvOperations.*;
import static br.com.guialves.rflr.gymnasium4j.utils.Numpy2DJLTypeMapper.numpyToDjl;

@Slf4j
@Accessors(fluent = true)
public class EnvProxy implements AutoCloseable {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;
    private final Cache<String, String> cache = Caffeine.newBuilder()
            .maximumSize(2)
            .build();
    private ZContext context;
    private SocketManager socket;
    private NDManager ndManager;
    private GymPythonLauncher launcher;
    private int flags;

    @Getter
    private EnvRenderMetadata renderMetadata;
    @Getter
    private EnvStateMetadata stateMetadata;
    private ByteBuffer imageBuffer;
    private ByteBuffer stateBuffer;

    // Used by Mock
    EnvProxy() {

    }

    protected EnvProxy(ZContext context, SocketManager socket, GymPythonLauncher launcher, NDManager ndManager) {
        this(context, socket, launcher, ndManager, 0);
    }

    @SneakyThrows
    EnvProxy(ZContext context,
             SocketManager socket,
             GymPythonLauncher launcher,
             NDManager ndManager,
             int flags) {
        if (!ndManager.isOpen()) throw new IllegalStateException("NDManager must be opened!");
        this.context = context;
        this.socket = socket;
        this.launcher = launcher;
        this.ndManager = ndManager.newSubManager();
        this.flags = flags;
    }

    public BufferedImage render() {
        RENDER.exec(socket);

        if (renderMetadata == null) {
            var json = socket.recvStr();
            renderMetadata = GSON.fromJson(json, EnvRenderMetadata.class);
            renderMetadata.init();
            prepareImageBuffer();
        }

        fillByteBuffer(imageBuffer);
        return toBufferedImage(imageBuffer, renderMetadata);
    }

    private void fillByteBuffer(ByteBuffer buffer) {
        buffer.clear();
        socket.recvByteBuffer(buffer, flags);
        buffer.position(0);
    }

    private void prepareImageBuffer() {
        imageBuffer = ByteBuffer.allocateDirect(renderMetadata.size());
        imageBuffer.order(ByteOrder.nativeOrder());
    }

    private BufferedImage toBufferedImage(
            ByteBuffer buffer,
            EnvRenderMetadata meta
    ) {
        return switch (meta.channels()) {
            case 1 -> ImageFromByteBuffer.byteBufferToGrayscaleImage(
                    buffer, meta.width(), meta.height());
            case 3 -> ImageFromByteBuffer.byteBufferToImage(
                    buffer, meta.width(), meta.height(), false);
            case 4 -> ImageFromByteBuffer.byteBufferToImage(
                    buffer, meta.width(), meta.height(), true);
            default -> throw new IllegalArgumentException("Unsupported channels");
        };
    }

    public String actionSpaceStr() {
        return cache.get("actionSpaceStr", _ -> {
            log.info("Getting actionSpaceStr from env...");
            ACTION_SPACE_STR.exec(socket);
            var json = socket.recvStr();
            var result = GSON.fromJson(json, ActionSpaceStr.class);
            return result.actionSpaceStr;
        });
    }

    public String observationSpaceStr() {
        return cache.get("observationSpaceStr", _ -> {
            log.info("Getting observationSpaceStr from env...");
            OBSERVATION_SPACE_STR.exec(socket);
            var json = socket.recvStr();
            var result = GSON.fromJson(json, ObservationSpaceStr.class);
            return result.observationSpaceStr;
        });
    }

    public int actionSpaceSample() {
        ACTION_SPACE_SAMPLE.exec(socket);
        var json = socket.recvStr();
        var result = GSON.fromJson(json, Action.class);
        return result.action;
    }

    public EnvStepResult step(int action) {
        if (null == stateBuffer) throw new IllegalStateException("You should call reset() first!");

        STEP.exec(socket);
        socket.sendJson(action);

        var json = socket.recvStr();
        var result = GSON.fromJson(json, EnvStepResult.class);

        fillByteBuffer(stateBuffer);
        var state = ndManager.create(stateBuffer, stateMetadata.djlShape(), stateMetadata.djlType);
        return result.state(state);
    }

    @SuppressWarnings("unchecked")
    public Pair<Map<String, Object>, NDArray> reset() {
        RESET.exec(socket);
        if (stateMetadata == null) {
            var json = socket.recvStr();
            stateMetadata = GSON.fromJson(json, EnvStateMetadata.class);
            stateMetadata.init();
            stateBuffer = ByteBuffer.allocate(stateMetadata.size());
            stateBuffer.order(ByteOrder.nativeOrder());
        }
        Map<String, Object> info = GSON.fromJson(socket.recvStr(), Map.class);
        fillByteBuffer(stateBuffer);

        var state = ndManager.create(stateBuffer, stateMetadata.djlShape(), stateMetadata.djlType);
        return new Pair<>(info, state);
    }

    @Override
    public void close() {
        CLOSE.exec(socket);
        Try.run(ndManager::close).onFailure(ex -> log.error("Error on ndManager.close(): {0}", ex));
        Try.run(socket::close).onFailure(ex -> log.error("Error on socket.close(): {0}", ex));
        Try.run(context::close).onFailure(ex -> log.error("Error on context.close(): {0}", ex));
        Try.run(launcher::close).onFailure(ex -> log.error("Error on launcher.close(): {0}", ex));
    }

    private record Action(int action) {}

    private record ActionSpaceStr(String actionSpaceStr) {}

    private record ObservationSpaceStr(String observationSpaceStr) {}

    @Accessors(fluent = true)
    public static class EnvRenderMetadata extends NumpyAdapter {

        public EnvRenderMetadata(int[] shape, String dtype) {
            super(shape, dtype);
        }

        public int height() {
            return shape[0];
        }

        public int width() {
            return shape[1];
        }

        public int channels() {
            return shape.length > 2 ? shape[2] : 1;
        }
    }

    @Accessors(fluent = true)
    public static class EnvStateMetadata extends NumpyAdapter {

        private Shape djlShape;
        private DataType djlType;

        public Shape djlShape() {
            return djlShape;
        }

        public int size() {
            return size;
        }

        @Override
        void init() {
            super.init();
            long[] longShape = Arrays.stream(shape).mapToLong(i -> i).toArray();
            this.djlShape = new Shape(longShape);
            this.djlType = numpyToDjl(dtype);
        }
    }

    private static class NumpyAdapter {

        // used by the Gson
        protected int[] shape;
        @Getter
        protected String dtype;
        @Getter
        protected int bytesPerElement;
        @Getter
        protected int size;

        NumpyAdapter() {

        }

        NumpyAdapter(int[] shape, String dtype) {
            this.shape = shape;
            this.dtype = dtype;
        }

        void init() {
            int elements = Arrays.stream(shape).reduce(1, Math::multiplyExact);
            this.bytesPerElement = Numpy2DJLTypeMapper.bytesPerElement(dtype);
            this.size = Math.multiplyExact(elements, bytesPerElement);
        }
    }
}
