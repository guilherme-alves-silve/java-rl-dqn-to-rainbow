package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

class FactorizedNoiseTest {

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
    void shouldMatchShapeInSampleNoise() {
        var expectedShape = new Shape(5);
        var noise = FactorizedNoise.sampleNoise(manager, 5);
        assertEquals(expectedShape, noise.getShape());
    }

    @Test
    void shouldMatchShapeInSampleNoiseOuter() {
        var expectedShapeWeight = new Shape(10, 5);
        var expectedShapeBias = new Shape(10);
        var noises = FactorizedNoise.sampleNoiseOuter(manager, 5, 10);
        var epsWeight = noises.epsWeight();
        var epsBias = noises.epsBias();
        assertEquals(expectedShapeWeight, epsWeight.getShape());
        assertEquals(expectedShapeBias, epsBias.getShape());
    }

    @Test
    void shouldApplySignSqrtAbsTransform() {
        var fixedInput = manager.create(new float[]{-4f, 9f, 0f, -0.25f, 16f});

        var mockManager = mock(NDManager.class);
        when(mockManager.randomNormal(0f, 1f, new Shape(5), DataType.FLOAT32))
                .thenReturn(fixedInput);

        var result = FactorizedNoise.sampleNoise(mockManager, 5);

        var expected = new float[]{-2f, 3f, 0f, -0.5f, 4f};

        assertArrayEquals(expected, result.toFloatArray(), DELTA);
        verify(mockManager).randomNormal(0f, 1f, new Shape(5), DataType.FLOAT32);
    }

    @Test
    void shouldPreserveEpsOutCorrelationBetweenWeightAndBias() {
        var fixedOut = manager.create(new float[]{4f, -9f});
        var fixedIn = manager.create(new float[]{1f, -4f, 0f});

        var mockManager = mock(NDManager.class);
        when(mockManager.randomNormal(0f, 1f, new Shape(2), DataType.FLOAT32))
                .thenReturn(fixedOut);
        when(mockManager.randomNormal(0f, 1f, new Shape(3), DataType.FLOAT32))
                .thenReturn(fixedIn);

        var noise = FactorizedNoise.sampleNoiseOuter(mockManager, 3, 2);

        var expectedEpsBias = new float[]{2f, -3f};
        var expectedEpsWeight = new float[][]{
                {2f, -4f, 0f},
                {-3f, 6f, 0f}
        };

        assertArrayEquals(expectedEpsBias, noise.epsBias().toFloatArray(), DELTA);
        assertArrayEquals(
                flatten(expectedEpsWeight),
                noise.epsWeight().toFloatArray(),
                DELTA
        );

        assertEquals(new Shape(2), noise.epsBias().getShape());
        assertEquals(new Shape(2, 3), noise.epsWeight().getShape());
    }

    private float[] flatten(float[][] matrix) {
        var result = new float[matrix.length * matrix[0].length];
        var idx = 0;
        for (var row : matrix) {
            for (var v : row) {
                result[idx++] = v;
            }
        }
        return result;
    }
}
