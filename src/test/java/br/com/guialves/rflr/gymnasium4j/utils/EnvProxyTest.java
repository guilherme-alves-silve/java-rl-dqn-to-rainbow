package br.com.guialves.rflr.gymnasium4j.utils;

import br.com.guialves.rflr.gymnasium4j.dto.EnvRenderMetadata;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Objects;

import static br.com.guialves.rflr.gymnasium4j.transform.EnvOperations.*;
import static org.mockito.Mockito.*;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class EnvProxyTest {

    @Mock
    private SocketManager socket;
    @InjectMocks
    private EnvProxy env;

    @Test
    public void shouldExecOpActionSpaceStr() {
        var expectedResult = "Discrete(2)";
        var json = "{\"actionSpaceStr\": \"Discrete(2)\"}";
        when(socket.recvStr()).thenReturn(json);

        var result = env.actionSpaceStr();
        verify(socket, times(1)).recvStr();
        verify(socket, times(1)).sendStr(ACTION_SPACE_STR.value());
        assertEquals(expectedResult, result);
    }

    @Test
    public void shouldExecOpObservationSpaceStr() {
        var expectedResult = "Box(4,)";
        var json = "{\"observationSpaceStr\": \"Box(4,)\"}";
        when(socket.recvStr()).thenReturn(json);

        var result = env.observationSpaceStr();
        verify(socket, times(1)).recvStr();
        verify(socket, times(1)).sendStr(OBSERVATION_SPACE_STR.value());
        assertEquals(expectedResult, result);
    }

    @Test
    public void shouldExecOpActionSpaceSample() {

        var expectedAction = 1;
        var json = "{\"action\": 1}";
        when(socket.recvStr()).thenReturn(json);

        int result = env.actionSpaceSample();

        verify(socket).sendStr(ACTION_SPACE_SAMPLE.value());
        assertEquals(expectedAction, result);
    }

    @Test
    @SneakyThrows
    public void shouldRender() {
        var metadata = new EnvRenderMetadata(new int[] {400, 600, 3}, "uint8");

        var uri = Objects.requireNonNull(getClass()
                        .getClassLoader()
                        .getResource("expected_cart_pole_img.bin"),
                            "expected_cart_pole_img.bin is null!")
                        .toURI();
        var imgBytes = Files.readAllBytes(Paths.get(uri));
        var expectedResult = ImageFromByteBuffer.byteBufferToImage(
                ByteBuffer.wrap(imgBytes),
                metadata.width(),
                metadata.height(),
                false);
        var json = "{\"shape\": [400, 600, 3], \"dtype\": \"uint8\"}";
        when(socket.recvStr()).thenReturn(json);

        doAnswer(invocation -> {
            ByteBuffer inputBuf = invocation.getArgument(0);
            inputBuf.put(imgBytes);
            return null;
        }).when(socket).recvByteBuffer(any(ByteBuffer.class), anyInt());

        var result1 = env.render();
        var result2 = env.render();
        verify(socket, times(1)).recvStr();
        verify(socket, times(2)).recvByteBuffer(any(ByteBuffer.class), anyInt());
        verify(socket, times(2)).sendStr(RENDER.value());
        assertImgEquals(expectedResult, result1);
        assertImgEquals(expectedResult, result2);
    }

    @Test
    void shouldStep() {

    }

    private void assertImgEquals(BufferedImage expected, BufferedImage actual) {
        assertEquals(expected.getWidth(), actual.getWidth());
        assertEquals(expected.getHeight(), actual.getHeight());
        assertEquals(expected.getType(), actual.getType());
        var expectedByteArray = ((DataBufferByte) expected.getRaster().getDataBuffer()).getData();
        var actualByteArray = ((DataBufferByte) actual.getRaster().getDataBuffer()).getData();
        assertArrayEquals(expectedByteArray, actualByteArray);
    }
}
