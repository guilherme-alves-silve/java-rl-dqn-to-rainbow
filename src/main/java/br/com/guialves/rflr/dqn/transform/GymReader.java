package br.com.guialves.rflr.dqn.transform;

import ai.djl.util.JsonUtils;
import br.com.guialves.rflr.dqn.dto.EnvRenderMetadata;
import br.com.guialves.rflr.dqn.utils.ImageFromByteBuffer;
import com.google.gson.Gson;
import lombok.Getter;
import lombok.experimental.Accessors;
import org.zeromq.ZMQ;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@Accessors(fluent = true)
public class GymReader {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;

    @Getter
    private ByteBuffer imageBuffer = null;
    @Getter
    private EnvRenderMetadata metadata;

    public BufferedImage getImage(ZMQ.Socket socket) {
        return getImage(socket, 0);
    }

    public BufferedImage getImage(ZMQ.Socket socket, int flags) {

        if (null == metadata) {
            var json = socket.recvStr();
            metadata = GSON.fromJson(json, EnvRenderMetadata.class);
            imageBuffer = ByteBuffer.allocateDirect(metadata.size());
            imageBuffer.order(ByteOrder.nativeOrder());
            imageBuffer.clear();
        }

        socket.recvByteBuffer(imageBuffer, flags);
        imageBuffer.flip();
        return toBufferedImage(imageBuffer, metadata.width(), metadata.height(), metadata.channels());
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
