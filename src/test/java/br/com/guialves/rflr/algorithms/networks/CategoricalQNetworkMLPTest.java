package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CategoricalQNetworkMLPTest {

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
    void testForwardSingleInput() {
        int observations = 4;
        int actions = 2;
        var expectedShape = new Shape(1, actions);
        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var input = manager.ones(new Shape(1, observations));
        var output = net.forward(input);
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testForwardBatchInput() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;
        var expectedShape = new Shape(batchSize, actions);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var output = net.forward(batch);
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testForwardDistSingleInput() {
        int observations = 4;
        int actions = 2;
        int atoms = 51;
        var expectedShape = new Shape(1, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var input = manager.ones(new Shape(1, observations));
        var output = net.forwardDist(new NDList(input));
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testForwardDistBatchInput() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;
        int atoms = 51;
        var expectedShape = new Shape(batchSize, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var output = net.forwardDist(new NDList(batch));
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testForwardDistSingleInputCustomAtoms() {
        int observations = 4;
        int actions = 2;
        int atoms = 20;
        float vMin = -5.0f;
        float vMax = +5.0f;
        var expectedShape = new Shape(1, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, atoms, vMin, vMax, manager);
        var input = manager.ones(new Shape(1, observations));
        var output = net.forwardDist(new NDList(input));
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testForwardDistBatchInputCustomAtoms() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;
        int atoms = 20;
        float vMin = -5.0f;
        float vMax = +5.0f;
        var expectedShape = new Shape(batchSize, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, atoms, vMin, vMax, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var output = net.forwardDist(new NDList(batch));
        assertEquals(expectedShape, output.getShape());
    }

    @Test
    void testDistributionSumsToOne() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;
        int atoms = 51;
        var expectedShape = new Shape(batchSize, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var output = net.forwardDist(new NDList(batch));
        assertEquals(expectedShape, output.getShape());

        var logSums = output.sum(new int[] {2});
        float[] logSumArr = logSums.toFloatArray();
        for (float logSum : logSumArr) {
            // -exp(ln(softmax(input))) = softmax(input)
            float sum = (float) -Math.expm1(logSum);
            assertTrue(Math.abs(sum - 1.0f) < DELTA,
                    "Distribution must sum to 1, was " + sum);
        }
    }

    @Test
    void testDistributionToQValue() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;
        int atoms = 51;
        var expectedShape = new Shape(batchSize, actions, atoms);

        @Cleanup var net = new CategoricalQNetworkMLP(observations, actions, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var dist = net.forwardDist(new NDList(batch));
        assertEquals(expectedShape, dist.getShape());

        var qValue = net.qValuesFromDist(dist);
        System.out.println(qValue);
    }
}
