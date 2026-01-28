package br.com.guialves.rflr.gymnasium4j.utils;

import lombok.NoArgsConstructor;
import lombok.SneakyThrows;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.nio.ByteBuffer;

@NoArgsConstructor
public class ImageFromByteBuffer {

    public static BufferedImage byteBufferToImage(ByteBuffer buffer, int width, int height, boolean hasAlpha) {
        int imageType = hasAlpha ? BufferedImage.TYPE_4BYTE_ABGR : BufferedImage.TYPE_3BYTE_BGR;
        var image = new BufferedImage(width, height, imageType);
        byte[] imageData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        buffer.get(imageData);
        return image;
    }

    public static BufferedImage byteBufferToGrayscaleImage(ByteBuffer buffer, int width, int height) {
        var image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        byte[] imageData = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        buffer.get(imageData);
        return image;
    }

    @SneakyThrows
    public static void saveImage(BufferedImage image, String filepath) {
        File output = new File(filepath);
        ImageIO.write(image, "png", output);
    }
}
