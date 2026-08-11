package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CategoricalCrossEntropyLossTest {

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
    void testUniformDistributionsShouldGiveLogAtoms() {
        // uniform log-softmax: log(1/51) = -3.9318...
        var logUniform = manager.full(new Shape(2, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        // uniform m summing to 1
        var mUniform = manager.full(new Shape(2, 1, ATOMS), 1.0f / ATOMS);
        var loss = new CategoricalCrossEntropyLoss();
        var out = loss.evaluate(new NDList(mUniform), new NDList(logUniform));
        float lossVal = out.mean().getFloat();

        assertEquals((float) Math.log(ATOMS), lossVal, DELTA,
                "Cross-entropy of two uniform distributions must be log(51) ≈ 3.93, got " + lossVal);
    }

    @Test
    void testSumOverAtomsNotOverBatch() {
        int batch = 10;
        var logP = manager.full(new Shape(batch, 1, ATOMS), (float) Math.log(1.0 / ATOMS));
        var pseudoProjectedBellman = manager.full(new Shape(batch, 1, ATOMS), 1.0f / ATOMS);
        var loss = new CategoricalCrossEntropyLoss();
        var out = loss.evaluate(new NDList(pseudoProjectedBellman), new NDList(logP));
        float lossVal = out.mean().getFloat();
        assertEquals((float) Math.log(ATOMS), lossVal, DELTA,
                "Loss must be log(51) independent of batch size, got " + lossVal);
    }
}
