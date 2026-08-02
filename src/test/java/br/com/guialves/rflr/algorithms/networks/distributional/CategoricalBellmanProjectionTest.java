package br.com.guialves.rflr.algorithms.networks.distributional;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CategoricalBellmanProjectionTest {

    private static final float DELTA = 1e-6f;
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
    void shouldGenerateZSupportCorrectly() {
        var catProj = new CategoricalBellmanProjection();

        var expectedZSupport = new float[]{-10.0f, -9.6f, -9.2f, -8.8f, -8.4f, -8.0f, -7.6f, -7.2f, -6.8f,
                -6.3999996f,-6.0f, -5.6f, -5.2f, -4.7999997f, -4.4f, -4.0f, -3.6f, -3.1999998f,
                -2.7999997f, -2.4f, -2.0f, -1.5999994f, -1.1999998f, -0.8000002f, -0.39999962f,
                0.0f, 0.40000057f, 0.8000002f, 1.1999998f, 1.6000004f, 2.0f, 2.4000006f,
                2.8000002f, 3.1999998f, 3.6000004f, 4.0f, 4.4000006f, 4.8f, 5.2f, 5.6000004f,
                6.0f, 6.3999996f, 6.800001f, 7.200001f, 7.6000004f, 8.0f, 8.4f,
                8.800001f, 9.200001f, 9.6f, 10.0f};

        assertArrayEquals(expectedZSupport, catProj.supportVectorZ(), DELTA);
    }

    @Test
    void shouldMakeBellmanProjectionWithCorrectShape() {
        int batchSize = 6;
        int actions = 4;
        int atoms = 51;
        // discount factor
        float gamma = 0.9f;
        var expectedShape = new Shape(batchSize, 1, atoms);
        var ndDist = manager.randomNormal(0f, 1f, new Shape(batchSize, actions * atoms), DataType.FLOAT32);
        var ndRewards = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -1.0f});
        var ndDones = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 1f});

        var catProj = new CategoricalBellmanProjection();
        var ndOutProj = catProj.project(ndDist, ndRewards, ndDones, gamma);

        assertEquals(expectedShape, ndOutProj.getShape());
    }
}
