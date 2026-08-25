package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CategoricalNLLPERLossTest {

    private static final int ATOMS = 51;
    private static final float DELTA = 1e-6f;

    private static NDManager manager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterAll
    static void shutdown() {
        manager.close();
    }

    @Test
    void shouldUniformDistributionsWithNonUniformWeights() {
        int batch = 3;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);

        // Different weights for each sample: [0.5, 1.0, 2.0]
        var weights = manager.create(new float[]{0.5f, 1.0f, 2.0f}, new Shape(batch, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        // Expected: mean of (weights * log(51)) = log(51) * mean(weights)
        float expected = (float) Math.log(ATOMS) * (0.5f + 1.0f + 2.0f) / batch;
        float actual = out.mean().getFloat();

        assertEquals(expected, actual, DELTA,
                "Loss must be weighted average of log(51), expected " + expected + " got " + actual);
    }

    @Test
    void shouldSumOverAtomsNotOverBatchWithUniformWeights() {
        int batch = 10;
        var logP = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var pseudoProjectedBellman = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(pseudoProjectedBellman), new NDList(logP));
        float lossVal = out.mean().getFloat();

        assertEquals((float) Math.log(ATOMS), lossVal, DELTA,
                "Loss must be log(51) independent of batch size with uniform weights, got " + lossVal);
    }

    @Test
    void shouldWeightedLossWithExtremeWeights() {
        int batch = 2;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);

        // Extreme weights: very small and very large
        var weights = manager.create(new float[]{1e-6f, 1e6f}, new Shape(batch, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        // Expected: log(51) * mean(weights) = log(51) * (1e-6 + 1e6) / 2
        float expected = (float) Math.log(ATOMS) * (1e-6f + 1e6f) / 2f;
        float actual = out.mean().getFloat();

        assertEquals(expected, actual, 1e-3f,
                "Loss must handle extreme weights correctly, expected " + expected + " got " + actual);
    }

    @Test
    void shouldWeightsShapeHandling() {
        int batch = 4;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);

        var weights1D = manager.ones(new Shape(batch));
        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights1D);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float lossVal1 = out.mean().getFloat();

        var weights2D = manager.ones(new Shape(batch, 1));
        loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights2D);
        out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float lossVal2 = out.mean().getFloat();

        var weights3D = manager.ones(new Shape(batch, 1, 1));
        loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights3D);
        out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float lossVal3 = out.mean().getFloat();

        float expected = (float) Math.log(ATOMS);
        assertEquals(expected, lossVal1, DELTA, "Weights shape (batch,) should work");
        assertEquals(expected, lossVal2, DELTA, "Weights shape (batch, 1) should work");
        assertEquals(expected, lossVal3, DELTA, "Weights shape (batch, 1, 1) should work");
    }

    @Test
    void shouldWeightsAreClearedAfterEvaluation() {
        var logUniform = manager.full(new Shape(1, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(1, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(1, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        assertThrows(IllegalStateException.class,
                () -> loss.evaluate(new NDList(mUniform), new NDList(logUniform)),
                "Weights must be cleared after evaluation and throw exception on next call");
    }

    @Test
    void shouldThrowsExceptionWhenWeightsNotSet() {
        var logUniform = manager.full(new Shape(1, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(1, 1, ATOMS), 1.0f / ATOMS);

        var loss = CategoricalNLLPERLoss.noneReduction();

        assertThrows(IllegalStateException.class,
                () -> loss.evaluate(new NDList(mUniform), new NDList(logUniform)),
                "Must throw exception when normISWeights not set");
    }

    @Test
    void shouldDifferentBatchSizes() {
        int[] batchSizes = {1, 2, 5, 10, 32};

        for (int batch : batchSizes) {
            var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
            var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
            var weights = manager.ones(new Shape(batch, 1, 1));

            var loss = CategoricalNLLPERLoss.noneReduction();
            loss.normISWeights(weights);
            var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));
            float lossVal = out.mean().getFloat();

            assertEquals((float) Math.log(ATOMS), lossVal, DELTA,
                    "Loss must work for batch size " + batch + ", got " + lossVal);
        }
    }

    @Test
    void shouldCompareWithBaseLossWithUniformWeights() {
        int batch = 5;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        // Base loss
        var baseLoss = new CategoricalNLLLoss();
        var baseOut = baseLoss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float baseVal = baseOut.mean().getFloat();

        // PER loss with uniform weights
        var perLoss = CategoricalNLLPERLoss.noneReduction();
        perLoss.normISWeights(weights);
        var perOut = perLoss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float perVal = perOut.mean().getFloat();

        assertEquals(baseVal, perVal, DELTA,
                "PER loss with uniform weights must equal base NLL loss");
    }

    @Test
    void shouldShapeOfOutput() {
        int batch = 3;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        var loss = new CategoricalNLLPERLoss(CategoricalNLLPERLoss.Reduction.MEAN);
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        var shape = out.getShape();
        assertEquals(0, shape.dimension(),
                "Output should be scalar (0D tensor), got shape " + shape);
    }

    @Test
    void shouldReturnBatchOneForNoneReduction() {
        int batch = 3;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        assertEquals(1, out.getShape().dimension(),
                "NONE reduction currently returns 2D (batch, 1); see bug note");
        assertEquals(batch, out.getShape().get(0),
                "First dim must be batch, got " + out.getShape());
    }

    @Test
    void shouldReturnSameShapeAsPERL2LossForNoneReduction() {
        int batch = 3;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        var loss = CategoricalNLLPERLoss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        assertEquals(new Shape(batch), out.getShape(),
                "Categorical and PER L2 losses must have matching NONE shapes");
    }

    @Test
    void shouldReturnSameShapeAsPERL2LossForMeanScalarReduction() {
        int batch = 3;
        var logUniform = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var mUniform = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var weights = manager.ones(new Shape(batch, 1, 1));

        var loss = new CategoricalNLLPERLoss(CategoricalNLLPERLoss.Reduction.MEAN);
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));

        assertEquals(new Shape(), out.getShape(),
                "Categorical and PER L2 losses must have matching MEAN shapes");
    }
}
