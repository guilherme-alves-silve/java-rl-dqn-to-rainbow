package br.com.guialves.rflr.dqn.transform;

import ai.djl.util.JsonUtils;
import br.com.guialves.rflr.dqn.dto.EnvRenderMetadata;
import br.com.guialves.rflr.dqn.utils.ImageFromByteBuffer;
import br.com.guialves.rflr.dqn.utils.SocketManager;
import com.google.gson.Gson;
import lombok.Getter;
import lombok.experimental.Accessors;
import lombok.extern.log4j.Log4j2;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@Log4j2
@Accessors(fluent = true)
public class GymReader {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;

    @Getter
    private ByteBuffer imageBuffer = null;
    @Getter
    private EnvRenderMetadata metadata;

    public BufferedImage getImage(SocketManager socket) {
        return getImage(socket, 0);
    }

    public BufferedImage getImage(SocketManager socket, int flags) {

        try {
            if (null == metadata) {
                var json = socket.recvStr();
                metadata = GSON.fromJson(json, EnvRenderMetadata.class);
                prepareBufferImg();
            }
            imageBuffer.clear();
            socket.recvByteBuffer(imageBuffer, flags);
            imageBuffer.position(0);
            return toBufferedImage(imageBuffer, metadata.width(), metadata.height(), metadata.channels());
        } catch (Exception ex) {
            log.error("Error: " + ex.getMessage());
            socket.reset();
            return getImage(socket, flags);
        }
    }

    private void prepareBufferImg() {
        imageBuffer = ByteBuffer.allocateDirect(metadata.size());
        imageBuffer.order(ByteOrder.nativeOrder());
    }

    private BufferedImage toBufferedImage(ByteBuffer imageBuffer, int width, int height, int channels) {
        if (channels == 1) {
            return ImageFromByteBuffer.byteBufferToGrayscaleImage(imageBuffer, width, height);
        }

        if (channels == 3) {
            return ImageFromByteBuffer.byteBufferToImage(imageBuffer, width, height, false);
        }

        if (channels == 4) {
            return ImageFromByteBuffer.byteBufferToImage(imageBuffer, width, height, true);
        }

        throw new IllegalArgumentException("Unsupported image type!");
    }
}
