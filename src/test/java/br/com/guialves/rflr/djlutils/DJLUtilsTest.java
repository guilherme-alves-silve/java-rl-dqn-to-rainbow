package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToFloat32;
import static br.com.guialves.rflr.djlutils.DJLUtils.djlMapToLong;
import static org.junit.jupiter.api.Assertions.*;

class DJLUtilsTest {

    public static final float DELTA = 1e-6f;
    private static final int BATCH_SIZE = 32;

    private static NDManager manager;
    private static NDManager testManager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
        testManager = subMgr(manager, "test-manager");
    }

    @AfterAll
    static void shutdown() {
        testManager.close();
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

    @Test
    void djlMapToLongShouldNotLeakIntoSystemManager() {
        var batch = createBatch(BATCH_SIZE);

        for (int i = 0; i < 5; ++i) {
            try (var sub = subMgr(testManager, "warmup-long-" + i)) {
                var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
                assertNotNull(actions);
            }
        }

        int afterWarmup = managedArrayCount(manager);
        assertTrue(afterWarmup > 0,
                "System manager must hold at least one resource after warmup");

        for (int i = 0; i < 50; ++i) {
            try (var sub = subMgr(testManager, "step-long-" + i)) {
                var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
                assertNotNull(actions);
            }
        }

        int afterSteps = managedArrayCount(manager);
        assertEquals(afterWarmup, afterSteps,
                "djlMapToLong must not leak into the system manager across calls. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
        int afterStepsTestMgr = managedArrayCount(testManager);
        assertTrue(afterWarmup >= afterStepsTestMgr,
                "djlMapToLong must not leak into the test manager across calls. "
                        + "before=" + afterWarmup + " after=" + afterStepsTestMgr
                        + " delta=" + (afterStepsTestMgr - afterWarmup));
    }

    @Test
    void djlMapToLongMustPreserveShapeAndOwnership() {
        var batch = createBatch(BATCH_SIZE);
        try (var sub = subMgr(testManager, "shape-long")) {
            var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
            assertEquals(new Shape(BATCH_SIZE, 1), actions.getShape(),
                    "Output shape must be (batchSize, 1)");
            assertEquals(sub, actions.getManager(),
                    "Returned array must be attached to the caller-supplied subManager "
                            + "so that closing the sub releases it. "
                            + "Actual manager=" + actions.getManager().getName());
        }
    }

    @Test
    void djlMapToFloat32ShouldNotLeakIntoSystemManager() {
        var batch = createBatch(BATCH_SIZE);

        for (int i = 0; i < 5; ++i) {
            try (var sub = subMgr(testManager, "warmup-f32-" + i)) {
                var rewards = djlMapToFloat32(sub, batch, Experience::reward);
                assertNotNull(rewards);
            }
        }

        int afterWarmup = managedArrayCount(manager);
        assertTrue(afterWarmup > 0,
                "System manager must hold at least one resource after warmup");

        for (int i = 0; i < 50; ++i) {
            try (var sub = subMgr(testManager, "step-f32-" + i)) {
                var rewards = djlMapToFloat32(sub, batch, Experience::reward);
                assertNotNull(rewards);
            }
        }

        int afterSteps = managedArrayCount(manager);
        assertEquals(afterWarmup, afterSteps,
                "djlMapToFloat32 must not leak into the system manager across calls. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
        int afterStepsTestMgr = managedArrayCount(testManager);
        assertTrue(afterWarmup >= afterStepsTestMgr,
                "djlMapToFloat32 must not leak into the test manager across calls. "
                        + "before=" + afterWarmup + " after=" + afterStepsTestMgr
                        + " delta=" + (afterStepsTestMgr - afterWarmup));
    }

    @Test
    void djlMapToFloat32ShouldNotLeakOnLargeBatch() {
        int largeBatch = 1024;
        var batch = createBatch(largeBatch);

        for (int i = 0; i < 5; ++i) {
            try (var sub = subMgr(testManager, "warmup-f32-large-" + i)) {
                var rewards = djlMapToFloat32(sub, batch, Experience::reward);
                assertNotNull(rewards);
            }
        }

        int afterWarmup = managedArrayCount(manager);

        for (int i = 0; i < 20; ++i) {
            try (var sub = subMgr(testManager, "step-f32-large-" + i)) {
                var rewards = djlMapToFloat32(sub, batch, Experience::reward);
                assertNotNull(rewards);
            }
        }

        int afterSteps = managedArrayCount(manager);
        assertEquals(afterWarmup, afterSteps,
                "djlMapToFloat32 on a " + largeBatch + "-size batch must not leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
        int afterStepsTestMgr = managedArrayCount(testManager);
        assertTrue(afterWarmup >= afterStepsTestMgr,
                "djlMapToFloat32 must not leak into the test manager across calls. "
                        + "before=" + afterWarmup + " after=" + afterStepsTestMgr
                        + " delta=" + (afterStepsTestMgr - afterWarmup));
    }

    @Test
    void djlMapToFloat32MustPreserveShapeAndDtype() {
        var batch = createBatch(BATCH_SIZE);
        try (var sub = subMgr(testManager, "shape-f32")) {
            var rewards = djlMapToFloat32(sub, batch, Experience::reward);
            assertEquals(new Shape(BATCH_SIZE, 1), rewards.getShape(),
                    "Output shape must be (batchSize, 1)");
            assertEquals(DataType.FLOAT32, rewards.getDataType(),
                    "Output dtype must be FLOAT32");
            assertEquals(sub, rewards.getManager(),
                    "Returned array must be attached to the caller-supplied subManager "
                            + "so that closing the sub releases it. "
                            + "Actual manager=" + rewards.getManager().getName());
        }
    }

    @Test
    void djlMapToFloat32MustPreserveValues() {
        var batch = createBatch(BATCH_SIZE);
        try (var sub = subMgr(testManager, "values-f32")) {
            var rewards = djlMapToFloat32(sub, batch, Experience::reward);
            var asFloats = DJLUtils.toFloatArray(rewards);
            assertEquals(BATCH_SIZE, asFloats.length);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                assertEquals((float) batch[i].reward(), asFloats[i], DELTA,
                        "Reward at index " + i + " must equal Experience.reward");
            }
        }
    }

    private Experience[] createBatch(int n) {
        var batch = new Experience[n];
        for (int i = 0; i < n; ++i) {
            batch[i] = ExperienceFixture.createRandomExperience(testManager, i);
        }
        return batch;
    }
}
