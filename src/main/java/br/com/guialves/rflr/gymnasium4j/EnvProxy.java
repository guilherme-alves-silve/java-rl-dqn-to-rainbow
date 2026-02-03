package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.utils.GymPythonLauncher;
import br.com.guialves.rflr.gymnasium4j.utils.ImageFromByteBuffer;
import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.gson.Gson;
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

@Slf4j
@Accessors(fluent = true)
public class EnvProxy implements AutoCloseable {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;
    private final Cache<String, String> cache = Caffeine.newBuilder()
            .maximumSize(2)
            .build();
    private ZContext context;
    private SocketManager socket;
    private GymPythonLauncher gymPythonLauncher;
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

    EnvProxy(ZContext context, SocketManager socket, GymPythonLauncher launcher, NDManager ndManager) {
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
        launcher.start();
    }

    public BufferedImage render() {
        RENDER.exec(socket);

        if (renderMetadata == null) {
            var json = socket.recvStr();
            renderMetadata = GSON.fromJson(json, EnvRenderMetadata.class);
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
        STEP.exec(socket);
        socket.sendJson(action);

        var json = socket.recvStr();
        var result = GSON.fromJson(json, EnvStepResult.class);

        fillByteBuffer(stateBuffer);
        result.state = ndManager.create(stateBuffer, stateMetadata.djlShape());
        return result;
    }

    @SuppressWarnings("unchecked")
    public Pair<Map<String, Object>, NDArray> reset() {
        RESET.exec(socket);
        if (stateMetadata == null) {
            var json = socket.recvStr();
            stateMetadata = GSON.fromJson(json, EnvStateMetadata.class);
            stateBuffer = ByteBuffer.allocateDirect(stateMetadata.size());
            stateBuffer.order(ByteOrder.nativeOrder());
        }
        Map<String, Object> info = GSON.fromJson(socket.recvStr(), Map.class);
        fillByteBuffer(stateBuffer);

        return new Pair<>(info, ndManager.create(stateBuffer, stateMetadata.djlShape()));
    }

    @Override
    public void close() {
        CLOSE.exec(socket);
        ndManager.close();
        context.close();
        launcher.close();
    }

    private record Action(int action) {}

    private record ActionSpaceStr(String actionSpaceStr) {}

    private record ObservationSpaceStr(String observationSpaceStr) {}

    @Getter
    @Accessors(fluent = true)
    @RequiredArgsConstructor
    public static class EnvStepResult {
        private final double reward;
        private final boolean term;
        private final boolean trunc;
        private final Map<String, Object> info;
        private NDArray state;

        public boolean done() {
            return term || trunc;
        }
    }

    public record EnvRenderMetadata(int[] shape, String dtype) {

        public int height() {
            return shape[0];
        }

        public int width() {
            return shape[1];
        }

        public int channels() {
            return shape.length > 2 ? shape[2] : 1;
        }

        public int size() {
            return Arrays.stream(shape).reduce(1, (a, b) -> a * b) * getBytesPerElement();
        }

        private int getBytesPerElement() {
            return dtype.contains("float") || dtype.contains("int32") ? 4 : 1;
        }
    }

    @Accessors(fluent = true)
    public static class EnvStateMetadata {
        @Getter
        private int[] shape;
        @Getter
        private String dtype;
        private Shape djlShape;

        public Shape djlShape() {
            if (djlShape == null) {
                // Converte int[] para long[] para o construtor do DJL Shape
                long[] longShape = Arrays.stream(shape).mapToLong(i -> i).toArray();
                djlShape = new Shape(longShape);
            }
            return djlShape;
        }

        public int size() {
            int elements = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
            return elements * (dtype.contains("float") || dtype.contains("int32") ? 4 : 1);
        }
    }
}
