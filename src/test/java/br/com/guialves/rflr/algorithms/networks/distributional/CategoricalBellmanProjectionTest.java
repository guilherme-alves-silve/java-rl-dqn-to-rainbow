package br.com.guialves.rflr.algorithms.networks.distributional;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection.*;
import static org.junit.jupiter.api.Assertions.*;

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

        assertArrayEquals(expectedZSupport, catProj.support(), DELTA);
    }

    @ParameterizedTest(name = "[{index}] {arguments}")
    @MethodSource("provideValidInputShapesRandomNormal")
    void shouldMakeBellmanProjectionWithCorrectShape(Shape validShape, boolean standard) {
        int batchSize = (int) validShape.get(0);
        int atoms = (int) validShape.getLastDimension();
        // discount factor
        float gamma = 0.9f;
        var expectedShape = new Shape(batchSize, 1, atoms);
        // just the selected actions must receive the Bellman projection.
        var probNextDist = manager.randomNormal(0f, 1f, validShape, DataType.FLOAT32);
        var rewards = manager.linspace(0f, 1f, batchSize);
        rewards.set(new NDIndex("-1"), -1f);
        var dones = manager.zeros(new Shape(batchSize), DataType.FLOAT32);
        dones.set(new NDIndex("-1"), 1f);

        var catProj = standard? new CategoricalBellmanProjection() :
                new CategoricalBellmanProjection((int) validShape.getLastDimension(), V_MIN, V_MAX);
        var massDistTarget = catProj.project(probNextDist, rewards, dones, gamma);

        assertEquals(expectedShape, massDistTarget.getShape());
    }

    @ParameterizedTest(name = "[{index}] {arguments}")
    @MethodSource("provideInvalidInputShapesRandomNormal")
    void shouldRefuseBatchWithMultipleActions(Shape invalidShape) {
        // discount factor
        float gamma = 0.9f;
        var probNextDist = manager.randomNormal(0f, 1f, invalidShape, DataType.FLOAT32);
        var rewards = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -1.0f});
        var dones = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 1f});

        var catProj = new CategoricalBellmanProjection();
        var exception = assertThrowsExactly(IllegalArgumentException.class, () -> catProj.project(probNextDist, rewards, dones, gamma));
        assertEquals("Invalid shape " + invalidShape + ", must be (batch, 1, %d) ".formatted(N_ATOMS),
                exception.getMessage());
    }

    @ParameterizedTest(name = "[{index}] {arguments}")
    @MethodSource("provideInvalidDimensionsShapesRandomNormal")
    void shouldRefuseBatchWithInvalidNumberOfDimensions(Shape invalidShapeDimensions, int expectedDim) {
        // discount factor
        float gamma = 0.9f;
        var probNextDist = manager.randomNormal(0f, 1f, invalidShapeDimensions, DataType.FLOAT32);
        var rewards = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -1.0f});
        var dones = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 1f});

        var catProj = new CategoricalBellmanProjection();
        var exception = assertThrowsExactly(IllegalArgumentException.class, () -> catProj.project(probNextDist, rewards, dones, gamma));
        assertEquals("Different dimensions between shape1 and shape2: %d != %d".formatted(
                invalidShapeDimensions.dimension(), expectedDim),
                exception.getMessage());
    }

    @Test
    void shouldRefuseInvalidRewardsAndDonesShape() {
        // discount factor
        var validShape = new Shape(6, 1, N_ATOMS);
        float gamma = 0.9f;
        var probNextDist = manager.randomNormal(0f, 1f, validShape, DataType.FLOAT32);
        var invalidRewards1 = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
        var dones1 = manager.create(new float[] {0f, 0f, 0f, 0f, 0f, 1f});

        var catProj = new CategoricalBellmanProjection();
        var exception1 = assertThrowsExactly(IllegalArgumentException.class, () -> catProj.project(probNextDist, invalidRewards1, dones1, gamma));
        assertEquals("Invalid dimensions between probNextDist (len=6), rewards (len=5) or dones (len=6)", exception1.getMessage());

        var rewards2 = manager.create(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, -1.0f});
        var invalidDones2 = manager.create(new float[] {0f, 0f, 0f, 0f, 0f});

        var exception2 = assertThrowsExactly(IllegalArgumentException.class, () -> catProj.project(probNextDist, rewards2, invalidDones2, gamma));
        assertEquals("Invalid dimensions between probNextDist (len=6), rewards (len=6) or dones (len=5)", exception2.getMessage());
    }

    static Stream<Arguments> provideValidInputShapesRandomNormal() {
        int action = 1;
        return Stream.of(
                Arguments.of(new Shape(75, action, 5), false, "distBatch75Action1Shape"),
                Arguments.of(new Shape(50, action, 10), false, "distBatch50Action1Shape"),
                Arguments.of(new Shape(30, action, 20), false, "distBatch30Action1Shape"),
                Arguments.of(new Shape(15, action, 30), false, "distBatch15Action1Shape"),
                Arguments.of(new Shape(10, action, 45), false, "distBatch10Action1Shape"),
                Arguments.of(new Shape(5, action, N_ATOMS), true, "distBatch5Action1Shape")
        );
    }

    static Stream<Arguments> provideInvalidDimensionsShapesRandomNormal() {
        int batchSize = 6;
        int action = 1;
        return Stream.of(
                Arguments.of(new Shape(batchSize * action * N_ATOMS), 3, "allFlatShape"),
                Arguments.of(new Shape(batchSize, action * N_ATOMS), 3, "flattenShape")
        );
    }

    static Stream<Arguments> provideInvalidInputShapesRandomNormal() {
        int batchSize = 6;
        int actions = 4;
        int differentAtoms = 20;
        return Stream.of(
                Arguments.of(new Shape(batchSize, actions, differentAtoms), "distActionsShape"),
                Arguments.of(new Shape(batchSize, actions, N_ATOMS), "distActionsShape")
        );
    }
}
