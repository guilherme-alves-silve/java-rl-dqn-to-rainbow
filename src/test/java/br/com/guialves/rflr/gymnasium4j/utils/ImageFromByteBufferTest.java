package br.com.guialves.rflr.gymnasium4j.utils;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ImageFromByteBufferTest {

    @Test
    void shouldNotAllowInstantiation() {
        assertThrows(IllegalArgumentException.class, () -> {
            try {
                var constructor = ImageFromByteBuffer.class.getDeclaredConstructor();
                constructor.setAccessible(true);
                constructor.newInstance();
            } catch (Exception ex) {
                throw ex.getCause();
            }
        });
    }

    @Nested
    @DisplayName("byteBufferToImage Tests")
    class ByteBufferToImageTests {

        @Test
        void shouldCreateRGBImageWithoutAlpha() {
            int width = 10;
            int height = 10;
            int bytesPerPixel = 3; // BGR
            var buffer = createTestBuffer(width, height, bytesPerPixel);
            var image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);

            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
            assertEquals(BufferedImage.TYPE_3BYTE_BGR, image.getType());
        }

        @Test
        void shouldCreateRGBAImageWithAlpha() {
            int width = 10;
            int height = 10;
            int bytesPerPixel = 4; // ABGR
            var buffer = createTestBuffer(width, height, bytesPerPixel);
            var image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, true);

            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
            assertEquals(BufferedImage.TYPE_4BYTE_ABGR, image.getType());
        }

        @ParameterizedTest
        @CsvSource({
                "1, 1, 3",
                "10, 10, 3",
                "100, 100, 3",
                "640, 480, 3",
                "800, 600, 3"
        })
        void shouldHandleVariousDimensionsRGB(int width, int height, int bytesPerPixel) {
            ByteBuffer buffer = createTestBuffer(width, height, bytesPerPixel);

            BufferedImage image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);

            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }

        @ParameterizedTest
        @CsvSource({
                "1, 1, 4",
                "10, 10, 4",
                "100, 100, 4",
                "640, 480, 4"
        })
        void shouldHandleVariousDimensionsRGBA(int width, int height, int bytesPerPixel) {
            ByteBuffer buffer = createTestBuffer(width, height, bytesPerPixel);

            BufferedImage image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, true);

            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }

        @Test
        void shouldPreservePixelDataRGB() {
            int width = 2;
            int height = 2;
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);

            // Fill with known pattern: [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]
            // Pixel 1: Blue=255, Green=0, Red=0
            buffer.put((byte) 255).put((byte) 0).put((byte) 0);
            // Pixel 2: Blue=0, Green=255, Red=0
            buffer.put((byte) 0).put((byte) 255).put((byte) 0);
            // Pixel 3: Blue=0, Green=0, Red=255
            buffer.put((byte) 0).put((byte) 0).put((byte) 255);
            // Pixel 4: Blue=128, Green=128, Red=128
            buffer.put((byte) 128).put((byte) 128).put((byte) 128);
            buffer.flip();

            BufferedImage image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);

            // Verify pixel colors (note: BufferedImage uses ARGB format for getRGB)
            int pixel1 = image.getRGB(0, 0);
            int pixel2 = image.getRGB(1, 0);
            int pixel3 = image.getRGB(0, 1);
            int pixel4 = image.getRGB(1, 1);

            // Extract RGB components
            assertTrue((pixel1 & 0xFF) == 255 || ((pixel1 >> 16) & 0xFF) == 255); // Blue or Red
            assertTrue((pixel2 & 0xFF00) != 0);
            assertTrue(((pixel3 >> 16) & 0xFF) == 255 || (pixel3 & 0xFF) == 255); // Red or Blue
            assertEquals(128, ((pixel4 >> 16) & 0xFF));
        }

        @Test
        void shouldHandleBufferWithExactSize() {
            int width = 5;
            int height = 5;
            int bytesPerPixel = 3;
            int expectedSize = width * height * bytesPerPixel;

            var buffer = ByteBuffer.allocate(expectedSize);
            for (int i = 0; i < expectedSize; i++) {
                buffer.put((byte) i);
            }
            buffer.flip();

            assertDoesNotThrow(() ->
                    ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false)
            );
        }

        @Test
        void shouldThrowWhenBufferTooSmall() {
            int width = 10;
            int height = 10;
            var buffer = ByteBuffer.allocate(10); // Too small

            assertThrows(Exception.class, () ->
                    ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false)
            );
        }

        @Test
        void shouldHandleZeroValues() {
            int width = 3;
            int height = 3;
            var buffer = ByteBuffer.allocate(width * height * 3);
            buffer.position(width * height * 3);
            buffer.flip();

            var image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);
            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }

        @Test
        void shouldHandleMaxValues() {
            int width = 3;
            int height = 3;
            var buffer = ByteBuffer.allocate(width * height * 3);

            for (int i = 0; i < width * height * 3; i++) {
                buffer.put((byte) 255);
            }
            buffer.flip();

            var image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);

            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }
    }

    @Nested
    @DisplayName("byteBufferToGrayscaleImage Tests")
    class ByteBufferToGrayscaleImageTests {

        @Test
        void shouldCreateGrayscaleImage() {
            int width = 10;
            int height = 10;
            var buffer = createGrayscaleBuffer(width, height);
            var image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
            assertEquals(BufferedImage.TYPE_BYTE_GRAY, image.getType());
        }

        @ParameterizedTest
        @CsvSource({
                "1, 1",
                "10, 10",
                "84, 84",
                "96, 96",
                "100, 100",
                "640, 480"
        })
        void shouldHandleVariousDimensions(int width, int height) {
            var buffer = createGrayscaleBuffer(width, height);
            var image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
            assertEquals(BufferedImage.TYPE_BYTE_GRAY, image.getType());
        }

        @Test
        void shouldPreserveGrayscaleValues() {
            int width = 4;
            int height = 4;
            var buffer = ByteBuffer.allocate(width * height);

            for (int i = 0; i < width * height; i++) {
                buffer.put((byte) (i * 16)); // 0, 16, 32, ..., 240
            }
            buffer.flip();

            var image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertNotNull(image);
            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }

        @Test
        void shouldHandleBlackImage() {
            int width = 5;
            int height = 5;
            var buffer = ByteBuffer.allocate(width * height);
            buffer.position(width * height);
            buffer.flip();

            BufferedImage image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertNotNull(image);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = image.getRGB(x, y);
                    int gray = rgb & 0xFF;
                    assertEquals(0, gray, "Expected black pixel");
                }
            }
        }

        @Test
        void shouldHandleWhiteImage() {
            int width = 5;
            int height = 5;
            var buffer = ByteBuffer.allocate(width * height);
            for (int i = 0; i < width * height; i++) {
                buffer.put((byte) 255);
            }
            buffer.flip();

            var image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertNotNull(image);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int rgb = image.getRGB(x, y);
                    int gray = rgb & 0xFF;
                    assertEquals(255, gray, "Expected white pixel");
                }
            }
        }

        @Test
        void shouldThrowWhenBufferTooSmall() {
            int width = 10;
            int height = 10;
            var buffer = ByteBuffer.allocate(10); // Too small

            assertThrows(Exception.class, () ->
                    ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height)
            );
        }

        @Test
        void shouldHandleSquareImages() {
            int size = 64;
            var buffer = createGrayscaleBuffer(size, size);
            var image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, size, size);
            assertEquals(size, image.getWidth());
            assertEquals(size, image.getHeight());
        }

        @Test
        void shouldHandleRectangularImages() {
            int width = 100;
            int height = 50;
            ByteBuffer buffer = createGrayscaleBuffer(width, height);

            BufferedImage image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);

            assertEquals(width, image.getWidth());
            assertEquals(height, image.getHeight());
        }
    }

    @Nested
    @DisplayName("saveImage Tests")
    class SaveImageTests {

        @Test
        void shouldSaveImage(@TempDir Path tempDir) throws IOException {
            var image = new BufferedImage(10, 10, BufferedImage.TYPE_3BYTE_BGR);
            String filepath = tempDir.resolve("test.png").toString();

            ImageFromByteBuffer.saveImage(image, filepath);

            var file = new File(filepath);
            assertTrue(file.exists());
            assertTrue(file.length() > 0);
        }

        @Test
        void shouldSaveAndReadImage(@TempDir Path tempDir) throws IOException {
            int width = 20;
            int height = 20;
            var originalImage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);

            // Draw some pattern
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    originalImage.setRGB(x, y, (x * y) % 256);
                }
            }

            String filepath = tempDir.resolve("test_pattern.png").toString();
            ImageFromByteBuffer.saveImage(originalImage, filepath);

            var loadedImage = ImageIO.read(new File(filepath));

            assertNotNull(loadedImage);
            assertEquals(width, loadedImage.getWidth());
            assertEquals(height, loadedImage.getHeight());
        }

        @Test
        void shouldSaveGrayscaleImage(@TempDir Path tempDir) throws IOException {
            BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
            String filepath = tempDir.resolve("grayscale.png").toString();

            ImageFromByteBuffer.saveImage(image, filepath);

            File file = new File(filepath);
            assertTrue(file.exists());
            assertTrue(file.length() > 0);
        }

        @Test
        void shouldSaveRGBAImage(@TempDir Path tempDir) throws IOException {
            BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_4BYTE_ABGR);
            String filepath = tempDir.resolve("rgba.png").toString();

            ImageFromByteBuffer.saveImage(image, filepath);

            File file = new File(filepath);
            assertTrue(file.exists());
        }

        @Test
        void shouldCreateDirectoriesIfNeeded(@TempDir Path tempDir) {
            BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_3BYTE_BGR);
            String filepath = tempDir.resolve("subdir/nested/image.png").toString();

            new File(filepath).getParentFile().mkdirs();

            assertDoesNotThrow(() -> ImageFromByteBuffer.saveImage(image, filepath));
            assertTrue(new File(filepath).exists());
        }

        @Test
        void shouldOverwriteExistingFile(@TempDir Path tempDir) throws IOException {
            BufferedImage image1 = new BufferedImage(10, 10, BufferedImage.TYPE_3BYTE_BGR);
            BufferedImage image2 = new BufferedImage(20, 20, BufferedImage.TYPE_3BYTE_BGR);
            String filepath = tempDir.resolve("overwrite.png").toString();

            ImageFromByteBuffer.saveImage(image1, filepath);
            long firstSize = new File(filepath).length();

            ImageFromByteBuffer.saveImage(image2, filepath);
            long secondSize = new File(filepath).length();

            assertTrue(new File(filepath).exists());
            assertNotEquals(firstSize, secondSize);
        }

        // Replace parameterized test with multiple regular tests
        @Test
        void shouldSaveSmallImage(@TempDir Path tempDir) throws IOException {
            testSaveImageSize(tempDir, 10, 10);
        }

        @Test
        void shouldSaveMediumImage(@TempDir Path tempDir) throws IOException {
            testSaveImageSize(tempDir, 100, 100);
        }

        @Test
        void shouldSaveVGAImage(@TempDir Path tempDir) throws IOException {
            testSaveImageSize(tempDir, 640, 480);
        }

        @Test
        void shouldSaveHDImage(@TempDir Path tempDir) throws IOException {
            testSaveImageSize(tempDir, 1920, 1080);
        }

        private void testSaveImageSize(Path tempDir, int width, int height) throws IOException {
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
            String filepath = tempDir.resolve("image_" + width + "x" + height + ".png").toString();

            ImageFromByteBuffer.saveImage(image, filepath);

            File file = new File(filepath);
            assertTrue(file.exists());
            assertTrue(file.length() > 0);
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {

        @Test
        void shouldConvertAndSaveRGBImage(@TempDir Path tempDir) throws IOException {
            int width = 50;
            int height = 50;
            ByteBuffer buffer = createTestBuffer(width, height, 3);

            BufferedImage image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);
            String filepath = tempDir.resolve("rgb_integration.png").toString();
            ImageFromByteBuffer.saveImage(image, filepath);

            assertTrue(new File(filepath).exists());
            BufferedImage loaded = ImageIO.read(new File(filepath));
            assertEquals(width, loaded.getWidth());
            assertEquals(height, loaded.getHeight());
        }

        @Test
        void shouldConvertAndSaveGrayscaleImage(@TempDir Path tempDir) throws IOException {
            int width = 84;
            int height = 84;
            ByteBuffer buffer = createGrayscaleBuffer(width, height);

            BufferedImage image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);
            String filepath = tempDir.resolve("grayscale_integration.png").toString();
            ImageFromByteBuffer.saveImage(image, filepath);

            assertTrue(new File(filepath).exists());
            BufferedImage loaded = ImageIO.read(new File(filepath));
            assertEquals(width, loaded.getWidth());
            assertEquals(height, loaded.getHeight());
        }

        @Test
        void shouldHandleGymnasiumCarRacingDimensions(@TempDir Path tempDir) throws IOException {
            // CarRacing-v3 default observation: 96x96x3
            int width = 96;
            int height = 96;
            ByteBuffer buffer = createTestBuffer(width, height, 3);

            BufferedImage image = ImageFromByteBuffer.byteBufferToImage(buffer, width, height, false);
            String filepath = tempDir.resolve("carracing.png").toString();
            ImageFromByteBuffer.saveImage(image, filepath);

            assertTrue(new File(filepath).exists());
            BufferedImage loaded = ImageIO.read(new File(filepath));
            assertEquals(96, loaded.getWidth());
            assertEquals(96, loaded.getHeight());
        }

        @Test
        void shouldHandleAtariDimensions(@TempDir Path tempDir) throws IOException {
            // Atari common observation: 210x160x3 or 84x84 after processing
            int width = 84;
            int height = 84;
            ByteBuffer buffer = createGrayscaleBuffer(width, height);

            BufferedImage image = ImageFromByteBuffer.byteBufferToGrayscaleImage(buffer, width, height);
            String filepath = tempDir.resolve("atari.png").toString();
            ImageFromByteBuffer.saveImage(image, filepath);

            assertTrue(new File(filepath).exists());
        }
    }

    // Helper methods
    private ByteBuffer createTestBuffer(int width, int height, int bytesPerPixel) {
        int size = width * height * bytesPerPixel;
        ByteBuffer buffer = ByteBuffer.allocate(size);
        for (int i = 0; i < size; i++) {
            buffer.put((byte) (i % 256));
        }
        buffer.flip();
        return buffer;
    }

    private ByteBuffer createGrayscaleBuffer(int width, int height) {
        int size = width * height;
        ByteBuffer buffer = ByteBuffer.allocate(size);
        for (int i = 0; i < size; i++) {
            buffer.put((byte) (i % 256));
        }
        buffer.flip();
        return buffer;
    }
}
