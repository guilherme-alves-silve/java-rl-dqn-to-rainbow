package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NoisyLayerTest {

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
    void shouldProduceExpectedOutputShapeInTraining() {
        var layer = new NoisyLayer(8);
        // batch=4, inFeatures=16
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var input = manager.randomNormal(inputShape);
        var paramStore = new ParameterStore(manager, false);

        var output = layer.forward(paramStore, new NDList(input), true).singletonOrThrow();

        assertEquals(new Shape(4, 8), output.getShape());
    }

    @Test
    void shouldProduceExpectedOutputShapeInInference() {
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var input = manager.randomNormal(inputShape);
        var paramStore = new ParameterStore(manager, false);

        var output = layer.forward(paramStore, new NDList(input), false).singletonOrThrow();

        assertEquals(new Shape(4, 8), output.getShape());
    }

    @Test
    void shouldBeDeterministicDuringInference() {
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var input = manager.randomNormal(inputShape);
        var paramStore = new ParameterStore(manager, false);

        var output1 = layer.forward(paramStore, new NDList(input), false).singletonOrThrow();
        var output2 = layer.forward(paramStore, new NDList(input), false).singletonOrThrow();

        assertArrayEquals(output1.toFloatArray(), output2.toFloatArray(), DELTA);
    }

    @Test
    void shouldDifferBetweenTrainingAndInferenceOutputs() {
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var input = manager.randomNormal(inputShape);
        var paramStore = new ParameterStore(manager, false);

        var trainOutput = layer.forward(paramStore, new NDList(input), true).singletonOrThrow();
        var inferOutput = layer.forward(paramStore, new NDList(input), false).singletonOrThrow();

        assertFalse(trainOutput.allClose(inferOutput));
    }

    @Test
    void shouldResampleNoiseWhenResetNoiseIsInvokedOnEachTrainingForward() {
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var input = manager.randomNormal(inputShape);
        var paramStore = new ParameterStore(manager, false);

        var output1 = layer.forward(paramStore, new NDList(input), true).singletonOrThrow();
        layer.resetNoise();
        var output2 = layer.forward(paramStore, new NDList(input), true).singletonOrThrow();

        assertFalse(output1.allClose(output2));
    }

    @Test
    void shouldInitializeParametersWithExpectedShapes() {
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var parameters = layer.getParameters();

        assertEquals(new Shape(8, 16), parameters.get("weightMu").getArray().getShape());
        assertEquals(new Shape(8, 16), parameters.get("weightSigma").getArray().getShape());
        assertEquals(new Shape(8), parameters.get("biasMu").getArray().getShape());
        assertEquals(new Shape(8), parameters.get("biasSigma").getArray().getShape());
    }

    @Test
    void shouldFillWeightSigmaWithExpectedConstantValue() {
        int batchSize = 4;
        var inFeatures = 16;
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(batchSize, inFeatures);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var weightSigma = layer.getParameters().get("weightSigma").getArray();
        var expected = 0.5f / (float) Math.sqrt(inFeatures);

        for (var value : weightSigma.toFloatArray()) {
            assertEquals(expected, value, DELTA);
        }
    }

    @Test
    void shouldFillBiasSigmaWithOutFeaturesDenominator() {
        var outFeatures = 8;
        var layer = new NoisyLayer(outFeatures);
        var inputShape = new Shape(4, 16);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var biasSigma = layer.getParameters().get("biasSigma").getArray();
        // asymmetry documented
        var expected = 0.5f / (float) Math.sqrt(outFeatures);

        for (var value : biasSigma.toFloatArray()) {
            assertEquals(expected, value, DELTA);
        }
    }

    @Test
    void shouldGenerateWeightMuWithinExpectedRange() {
        var inFeatures = 16;
        var layer = new NoisyLayer(8);
        var inputShape = new Shape(4, inFeatures);
        layer.initialize(manager, DataType.FLOAT32, inputShape);

        var weightMu = layer.getParameters().get("weightMu").getArray();
        var bound = 1f / (float) Math.sqrt(inFeatures);

        for (var value : weightMu.toFloatArray()) {
            assertTrue(value >= -bound && value <= bound,
                    "value " + value + " out of expected range [-" + bound + ", " + bound + "]");
        }
    }
}
