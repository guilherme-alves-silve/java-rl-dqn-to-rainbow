package br.com.guialves.rflr.algorithms.dqnper;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PERL2LossTest {

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
    void shouldZeroLossForPerfectPredictions() {
        int batch = 4;
        var pred = manager.create(new float[]{1f, 2f, 3f, 4f}, new Shape(batch, 1));
        var label = manager.create(new float[]{1f, 2f, 3f, 4f}, new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(0f, out.mean().getFloat(), DELTA,
                "Perfect predictions must yield zero loss");
    }

    @Test
    void shouldComputeKnownL2WithUniformWeights() {
        // pred = [1, 2], label = [2, 4]
        // error = [-1, -2], squared = [1, 4]
        // weighted (w=1) = [1, 4], * 0.5 = [0.5, 2.0]
        // mean = 1.25
        int batch = 2;
        var pred = manager.create(new float[]{1f, 2f}, new Shape(batch, 1));
        var label = manager.create(new float[]{2f, 4f}, new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(1.25f, out.mean().getFloat(), DELTA,
                "L2 loss with uniform weights must match hand-computed value");
    }

    @Test
    void shouldApplyWeightsCorrectly() {
        // Single sample, weight = 2.0, error = -1 -> squared=1, *2 *0.5 = 1.0
        int batch = 1;
        var pred = manager.create(new float[]{1f}, new Shape(batch, 1));
        var label = manager.create(new float[]{2f}, new Shape(batch, 1));
        var weights = manager.create(new float[]{2f}, new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(1.0f, out.mean().getFloat(), DELTA,
                "IS weight must scale the squared error by w * 0.5");
    }

    @Test
    void shouldNotApplyTheHalfOutsideTheWeight() {
        // If w=1, error=2 -> squared=4, *0.5 = 2.0
        // This test guards against a regression where the 0.5 is applied
        // twice or not at all.
        int batch = 1;
        var pred = manager.create(new float[]{0f}, new Shape(batch, 1));
        var label = manager.create(new float[]{2f}, new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(2.0f, out.mean().getFloat(), DELTA,
                "For w=1, error=2 the loss must be 0.5 * 1 * 4 = 2.0");
    }

    @Test
    void shouldMatchPlainL2WithUniformWeightsAcrossBatchSizes() {
        for (int batch : new int[]{1, 2, 8, 32}) {
            var pred = manager.create(new float[batch], new Shape(batch, 1))
                    .add(0.5f);   // 0.5 everywhere
            var label = manager.zeros(new Shape(batch, 1));
            var weights = manager.ones(new Shape(batch, 1));

            var loss = PERL2Loss.noneReduction();
            loss.normISWeights(weights);
            var out = loss.evaluate(new NDList(label), new NDList(pred));

            // 0.5 * 0.25 = 0.125
            assertEquals(0.125f, out.mean().getFloat(), DELTA,
                    "Loss must be invariant to batch size when w=1 and pred is constant; batch=" + batch);
        }
    }

    @Test
    void shouldReturnScalarForMeanReduction() {
        int batch = 3;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = new PERL2Loss(PERL2Loss.Reduction.MEAN);
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(0, out.getShape().dimension(),
                "MEAN reduction must return a 0D scalar, got " + out.getShape());
    }

    @Test
    void shouldReturnBatchOneForNoneReduction() {
        int batch = 3;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(1, out.getShape().dimension(),
                "NONE reduction must return 2D, got " + out.getShape());
    }

    @Test
    void shouldHaveDimensionOneForAll() {
        for (int batch : new int[]{2, 4, 16, 64}) {
            var pred = manager.zeros(new Shape(batch, 1));
            var label = manager.ones(new Shape(batch, 1));
            var weights = manager.ones(new Shape(batch, 1));

            var loss = PERL2Loss.noneReduction();
            loss.normISWeights(weights);
            var out = loss.evaluate(new NDList(label), new NDList(pred));

            assertEquals(new Shape(batch), out.getShape());
        }
    }

    @Test
    void shouldAcceptOneDimensionalWeights() {
        int batch = 4;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));
        var weights1D = manager.ones(new Shape(batch));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights1D);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        // 0.5 * 1 * 1 = 0.5
        assertEquals(0.5f, out.mean().getFloat(), DELTA,
                "1D weights must be reshaped to (batch, 1) and applied");
    }

    @Test
    void shouldAcceptTwoDimensionalWeightsAsIs() {
        int batch = 4;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));
        var weights2D = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights2D);
        var out = loss.evaluate(new NDList(label), new NDList(pred));

        assertEquals(0.5f, out.mean().getFloat(), DELTA,
                "2D (batch, 1) weights must be applied without reshape");
    }

    @Test
    void shouldThrowWhenWeightsNotSet() {
        int batch = 2;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();

        assertThrows(IllegalStateException.class,
                () -> loss.evaluate(new NDList(label), new NDList(pred)),
                "Must throw if normISWeights not set before evaluate");
    }

    @Test
    void shouldClearWeightsAfterEvaluation() {
        int batch = 2;
        var pred = manager.zeros(new Shape(batch, 1));
        var label = manager.ones(new Shape(batch, 1));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        loss.evaluate(new NDList(label), new NDList(pred));

        assertThrows(IllegalStateException.class,
                () -> loss.evaluate(new NDList(label), new NDList(pred)),
                "Second evaluate without re-setting weights must throw");
    }

    @Test
    void shouldReshapeLabelToMatchPred() {
        int batch = 3;
        var pred = manager.zeros(new Shape(batch, 1));
        var label1D = manager.ones(new Shape(batch));
        var weights = manager.ones(new Shape(batch, 1));

        var loss = PERL2Loss.noneReduction();
        loss.normISWeights(weights);
        var out = loss.evaluate(new NDList(label1D), new NDList(pred));

        // 0.5 * 1 * 1 = 0.5
        assertEquals(0.5f, out.mean().getFloat(), DELTA,
                "1D label must be reshaped to match pred shape");
    }
}
