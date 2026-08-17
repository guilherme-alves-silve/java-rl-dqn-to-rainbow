package br.com.guialves.rflr.algorithms.networks.layers.activation;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RReLUTest {

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
    void shouldNotModifyOutputShapes() {
        int batch = 5;
        int values = 20;

        var rrelu = new RReLU();
        var expectedOutputShape = new Shape[] { new Shape(batch, values) };

        var outputShape = rrelu.getOutputShapes(new Shape[] { new Shape(batch, values) });
        assertArrayEquals(expectedOutputShape, outputShape);
    }

    @Test
    void shouldApplyLeakRReLUWhenTraining() {
        var rrelu = new RReLU();
        var input = new float[] {-0.5f, -0.1f, 0.0f, 0.5f, 0.75f, 0.9f};
        var parameterStore = new ParameterStore();
        var inputs = new NDList() {{
            add(manager.create(input));
        }};
        boolean training = true;
        var params = new PairList<String, Object>();

        var out = rrelu.forwardInternal(parameterStore, inputs, training, params).singletonOrThrow();
        var outArr = out.toFloatArray();
        assertEquals(input.length, outArr.length);
        // Example output = [ -0.14072676, -0.013909191, 0.0, 0.5, 0.75, 0.9 ]
        assertTrue(outArr[0] > input[0]);
        assertTrue(outArr[1] > input[1]);
        assertEquals(input[2], outArr[2], DELTA);
        assertEquals(input[3], outArr[3], DELTA);
        assertEquals(input[4], outArr[4], DELTA);
        assertEquals(input[5], outArr[5], DELTA);
    }

    @Test
    void shouldRReLUWhenNotTrainingApplyDeterministicBehavior() {
        var rrelu = new RReLU();
        var input = new float[] {-0.5f, -0.1f, 0.0f, 0.5f, 0.75f, 0.9f};
        var expectedOutput = new float[] {-0.114583336f, -0.022916667f, 0.0f, 0.5f, 0.75f, 0.9f};
        var parameterStore = new ParameterStore();
        var inputs = new NDList() {{
            add(manager.create(input));
        }};
        boolean training = false;
        var params = new PairList<String, Object>();

        var out = rrelu.forwardInternal(parameterStore, inputs, training, params).singletonOrThrow();
        var outArr = out.toFloatArray();
        assertEquals(input.length, outArr.length);
        // Deterministic output = [ -0.114583336, -0.022916667, 0.0, 0.5, 0.75, 0.9 ]
        assertArrayEquals(expectedOutput, outArr, DELTA);
    }
}
