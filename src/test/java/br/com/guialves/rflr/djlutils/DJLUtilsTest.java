package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.debugDump;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static org.junit.jupiter.api.Assertions.*;

class DJLUtilsTest {

    public static final float DELTA = 1e-6f;

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
    void shouldCopyBlockParameters() {
        var net1 = new SequentialBlock()
                .add(Linear.builder()
                        .setUnits(128)
                        .optBias(true)
                        .build())
                .add(Activation::relu);
        net1.initialize(manager, DataType.FLOAT32, new Shape(2, 2));

        var net2 = new SequentialBlock()
                .add(Linear.builder()
                        .setUnits(128)
                        .optBias(true)
                        .build())
                .add(Activation::relu);
        net2.initialize(manager, DataType.FLOAT32, new Shape(2, 2));

        assertTrue(DJLUtils.diff(net1, net2));
        DJLUtils.copy(net1, net2);
        assertFalse(DJLUtils.diff(net1, net2));
    }

    @Test
    void shouldNotMemoryLeakWhenCallingToFloatArray() {
        int size = 1024;
        var array = manager.randomUniform(0f, 1f, new Shape(size), DataType.FLOAT32);

        // Warmup
        for (int i = 0; i < 5; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(size, result.length, "toFloatArray must return array of correct size");
        }

        int afterWarmup = managedArrayCount(manager);
        assertTrue(afterWarmup > 0, "Warmup must at least allocate one input array");
        debugDump(manager);

        // 50 iterações
        for (int i = 0; i < 50; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(size, result.length);
        }
        int afterSteps = managedArrayCount(manager);

        debugDump(manager);
        assertEquals(afterWarmup, afterSteps,
                "toFloatArray (with sub-manager) cannot leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
    }

    @Test
    void shouldNotMemoryLeakWhenCallingGetFloat() {
        int size = 1024;
        var scalar = manager.randomUniform(0f, 1f, new Shape(size), DataType.FLOAT32).sum();

        // Warmup
        for (int i = 0; i < 5; i++) {
            float result = DJLUtils.getFloat(scalar);
            assertTrue(Float.isFinite(result), "getFloat must has finite value.");
        }

        int afterWarmup = managedArrayCount(manager);
        assertTrue(afterWarmup > 0);
        debugDump(manager);

        for (int i = 0; i < 50; i++) {
            float result = DJLUtils.getFloat(scalar);
            assertTrue(Float.isFinite(result));
        }
        int afterSteps = managedArrayCount(manager);

        debugDump(manager);
        assertEquals(afterWarmup, afterSteps,
                "getFloat (com sub-manager) não pode vazar. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
    }

    @Test
    void shouldNotMemoryLeakOnLargeArray() {
        // Edge case: big array
        int size = 1_000_000;
        var array = manager.randomUniform(0f, 1f, new Shape(size), DataType.FLOAT32);

        for (int i = 0; i < 3; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(size, result.length);
        }
        int afterWarmup = managedArrayCount(manager);
        debugDump(manager);

        for (int i = 0; i < 10; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(size, result.length);
        }
        int afterSteps = managedArrayCount(manager);

        debugDump(manager);
        assertEquals(afterWarmup, afterSteps,
                "toFloatArray in big array cannot leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps);
    }

    @Test
    void shouldNotMemoryLeakOnMultiDimArray() {
        // Edge case: multi-dimensional array, sync GPU -> CPU can have different paths
        var array = manager.randomUniform(0f, 1f, new Shape(64, 32, 16), DataType.FLOAT32);

        for (int i = 0; i < 5; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(64 * 32 * 16, result.length);
        }
        int afterWarmup = managedArrayCount(manager);
        debugDump(manager);

        for (int i = 0; i < 20; i++) {
            float[] result = DJLUtils.toFloatArray(array);
            assertEquals(64 * 32 * 16, result.length);
        }
        int afterSteps = managedArrayCount(manager);

        debugDump(manager);
        assertEquals(afterWarmup, afterSteps,
                "toFloatArray in 3D array cannot leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps);
    }

    @Test
    void getFloatOnScalarShouldNotLeak() {
        // Edge case: array scalar (0-dim)
        var array = manager.create(42.0f);

        for (int i = 0; i < 5; i++) {
            float result = DJLUtils.getFloat(array);
            assertEquals(42.0f, result, DELTA);
        }
        int afterWarmup = managedArrayCount(manager);
        debugDump(manager);

        for (int i = 0; i < 20; i++) {
            float result = DJLUtils.getFloat(array);
            assertEquals(42.0f, result, DELTA);
        }
        int afterSteps = managedArrayCount(manager);

        debugDump(manager);
        assertEquals(afterWarmup, afterSteps,
                "getFloat in scalar array cannot leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps);
    }
}
