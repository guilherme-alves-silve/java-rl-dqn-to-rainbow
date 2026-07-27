package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class NoisyLayerInitTest {

    public static final float DELTA = 1e-6f;
    private static NDManager manager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterAll
    static void cleanup() {
        manager.close();
    }

    @Test
    void shouldCreateMuInitWithCorrectSizeAndType() {
        var init = NoisyLayerInit.ofMu(10);

        assertEquals(10, init.getSize());
        assertEquals(NoisyLayerInit.InitType.MU, init.getType());
    }

    @Test
    void shouldCreateSigmaInitWithCorrectSizeAndType() {
        var init = NoisyLayerInit.ofSigma(10);

        assertEquals(10, init.getSize());
        assertEquals(NoisyLayerInit.InitType.SIGMA, init.getType());
    }

    @Test
    void shouldFillSigmaWithExpectedConstantValue() {
        var size = 16;
        var init = NoisyLayerInit.ofSigma(size);
        var shape = new Shape(4, 4);

        var result = init.initialize(manager, shape, DataType.FLOAT32);

        var expected = 0.5f / (float) Math.sqrt(size);
        var values = result.toFloatArray();

        assertEquals(shape, result.getShape());
        for (var value : values) {
            assertEquals(expected, value, DELTA);
        }
    }

    @Test
    void shouldGenerateMuValuesWithinExpectedRange() {
        var size = 25;
        var init = NoisyLayerInit.ofMu(size);
        var shape = new Shape(100);

        var result = init.initialize(manager, shape, DataType.FLOAT32);

        // mu ~ U(-1/sqrt(size), 1/sqrt(size))
        var bound = 1f / (float) Math.sqrt(size);
        var values = result.toFloatArray();

        assertEquals(shape, result.getShape());
        for (var value : values) {
            assertTrue(value >= -bound && value <= bound,
                    "value " + value + " out of expected range [-" + bound + ", " + bound + "]");
        }
    }

    @Test
    void shouldCallRandomUniformWithCorrectBoundsForMu() {
        var size = 4;
        var init = NoisyLayerInit.ofMu(size);
        var shape = new Shape(2, 2);
        var expectedArray = manager.zeros(shape);

        var mockManager = mock(NDManager.class);
        when(mockManager.randomUniform(-0.5f, 0.5f, shape, DataType.FLOAT32))
                .thenReturn(expectedArray);

        var result = init.initialize(mockManager, shape, DataType.FLOAT32);

        assertSame(expectedArray, result);
        verify(mockManager).randomUniform(-0.5f, 0.5f, shape, DataType.FLOAT32);
    }

    @Test
    void shouldCallFullWithCorrectValueForSigma() {
        var size = 4;
        var init = NoisyLayerInit.ofSigma(size);
        var shape = new Shape(2, 2);
        var expectedArray = manager.zeros(shape);

        var mockManager = mock(NDManager.class);
        when(mockManager.full(shape, 0.25f, DataType.FLOAT32))
                .thenReturn(expectedArray);

        var result = init.initialize(mockManager, shape, DataType.FLOAT32);

        assertSame(expectedArray, result);
        verify(mockManager).full(shape, 0.25f, DataType.FLOAT32);
    }
}
