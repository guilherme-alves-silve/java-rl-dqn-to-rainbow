package br.com.guialves.rflr.gymnasium4j.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.JsonUtils;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.gson.Gson;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;
import lombok.extern.log4j.Log4j2;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;

import static br.com.guialves.rflr.gymnasium4j.transform.EnvOperations.*;

@Log4j2
@Accessors(fluent = true)
public class EnvProxy implements AutoCloseable {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;
    private static final Cache<String, String> CACHE_STR = Caffeine.newBuilder()
            .maximumSize(1)
            .build();

    private SocketManager socket;
    private NDManager ndManager;
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

    EnvProxy(SocketManager socket, NDManager ndManager) {
        this(socket, ndManager, 0);
    }

    EnvProxy(SocketManager socket, NDManager ndManager, int flags) {
        if (!ndManager.isOpen()) throw new IllegalStateException("NDManager must be opened!");
        this.socket = socket;
        this.ndManager = ndManager.newSubManager();
        this.flags = flags;
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
        return CACHE_STR.get("actionSpaceStr", _ -> {
            ACTION_SPACE_STR.exec(socket);
            var json = socket.recvStr();
            var result = GSON.fromJson(json, ActionSpaceStr.class);
            return result.actionSpaceStr;
        });
    }

    public String observationSpaceStr() {
        return CACHE_STR.get("observationSpaceStr", _ -> {
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

    public void reset() {
        RESET.exec(socket);

        var metaJson = socket.recvStr();
        var stateMeta = GSON.fromJson(metaJson, EnvStateMetadata.class);

        // info
        socket.recvStr();

        // state buffer
        //socket.recvByteBuffer();
        // TODO Implementar
    }

    @Override
    public void close() {
        CLOSE.exec(socket);
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
            return shape[2];
        }

        public int size() {
            return Arrays.stream(shape).reduce(1, ((left, right) -> left * right));
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
            if (null == djlShape) {
                djlShape = new Shape(shape[0], shape[1]);
            }

            return djlShape;
        }
    }
}
