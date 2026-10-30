package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Memory-leak regression tests for
 * {@link DJLMemoryManagement#safeForwardSingleFloat(NDManager, ai.djl.nn.Block, ParameterStore, NDList, boolean)}.
 *
 * <h2>What is under test</h2>
 * <p>{@code safeForwardSingleFloat} is the entry point used by
 * {@code DeepQNetworkCNN.forward} to run a forward pass on a single input head. Internally it:
 * <ol>
 *   <li>opens a short-lived sub-manager ({@code "safe-forward-single-*"});</li>
 *   <li>temporarily attaches the caller input to it;</li>
 *   <li>optionally converts the input to {@code FLOAT32} and rescales it by {@code 1/255};</li>
 *   <li>runs the block forward pass;</li>
 *   <li>re-attaches the output to the caller manager and closes the sub-manager.</li>
 * </ol>
 *
 * <h2>How a leak is detected</h2>
 * <p>Three independent signals are used, because each one catches a different class of defect:
 * <ul>
 *   <li><b>Resource growth</b> — {@link DJLMemoryManagement#managedArrayCount} and
 *       {@link DJLMemoryManagement#systemResourceCount} walk the whole manager hierarchy.
 *       A stable count across many calls means no NDArray and no sub-manager is retained.</li>
 *   <li><b>Dangling sub-managers</b> — the internal {@code "safe-forward-single-*"} sub-manager must
 *       never survive the call. A leftover node means the internal scope was not closed.</li>
 *   <li><b>Released intermediates</b> — a {@link LambdaBlock} placed at the head of the network
 *       captures the array the block actually consumed. In the byte-input path that array is an
 *       internal intermediate, and it must be <i>released</i> (native memory freed) once the call
 *       returns. This is the only signal that catches a leak where an array is unregistered from the
 *       manager but never closed, which no counter can observe.</li>
 * </ul>
 *
 * <h2>Ownership contract</h2>
 * <p>The returned {@link NDList} belongs to the caller and must be attached to the manager passed
 * in. The caller input must survive the call and be returned to its original manager, otherwise
 * the method would silently invalidate the caller's buffer.
 */
@DisplayName("Memory leak regression tests for DJLMemoryManagement")
class DJLMemoryManagementTest {

    /**
     * Prefix of the internal sub-manager created by {@code safeForwardSingleFloat}.
     * {@code DJLMemoryManagement.subMgr} names it {@code "<name>-<uid>"}.
     */
    private static final String INTERNAL_SUB_MANAGER_PREFIX = "safe-forward-single-";

    private static final int FEATURES = 4;
    private static final int HIDDEN = 8;
    private static final int ACTIONS = 3;

    private static final Shape INPUT_SHAPE = new Shape(1, FEATURES);
    private static final Shape OUTPUT_SHAPE = new Shape(1, ACTIONS);

    /** Raw byte input, used to exercise the UINT8 -> FLOAT32 conversion path. */
    private static final byte[] BYTE_INPUT = new byte[]{0, 64, (byte) 128, (byte) 255};

    /** Calls executed before the baseline is captured (JIT warm-up, lazy engine init). */
    private static final int WARMUP = 5;

    /**
     * Calls executed after the baseline is captured. The higher the number, the slower a
     * leak is detected: a leak of one array every N calls only shows up as a delta once
     * the loop runs more than N times.
     */
    private static final int ITERATIONS = 200;

    /** Tolerance for the coarse Java-heap retention check. */
    private static final long MAX_HEAP_GROWTH_BYTES = 64L * 1024 * 1024;

    private NDManager root;
    private NDManager netManager;
    private SequentialBlock net;
    private ParameterStore parameterStore;

    /** Enables the capture of the array consumed by the network. */
    private final AtomicBoolean recording = new AtomicBoolean(false);

    /** Holds the array the network actually received during the last recorded call. */
    private final AtomicReference<NDArray> capturedInput = new AtomicReference<>();

    /**
     * Holds the data type of the captured array, read while the array was still alive.
     * It cannot be read afterwards because the array is released by the internal
     * sub-manager when the call returns.
     */
    private final AtomicReference<DataType> capturedDataType = new AtomicReference<>();

    @BeforeEach
    @DisplayName("Build a small MLP under a dedicated sub-manager")
    void setUp() {
        root = NDManager.newBaseManager();
        netManager = root.newSubManager();
        netManager.setName("net-" + netManager.getName());

        // The LambdaBlock sits at the head of the network so it can capture the exact
        // array the block consumed, including the internal FLOAT32 intermediate
        // produced by the byte -> float conversion.
        var capture = new LambdaBlock(inputs -> {
            if (recording.get()) {
                var array = inputs.singletonOrThrow();
                capturedInput.set(array);
                capturedDataType.set(array.getDataType());
            }
            return inputs;
        }, "capture");

        net = new SequentialBlock();
        net.add(capture)
                .add(DJLLayers.linear(HIDDEN))
                .add(Activation::relu)
                .add(DJLLayers.linear(ACTIONS));
        net.initialize(netManager, DataType.FLOAT32, INPUT_SHAPE);

        parameterStore = new ParameterStore(netManager, false);
    }

    @AfterEach
    @DisplayName("Release the manager tree")
    void tearDown() {
        recording.set(false);
        root.close();
    }

    @Test
    @DisplayName("Repeated FLOAT32 calls must not grow the manager resource tree")
    void shouldNotLeakResourcesOnRepeatedFloat32Calls() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);

        assertNoResourceGrowth(input, "FLOAT32");
    }

    @Test
    @DisplayName("Repeated byte calls must not grow the manager resource tree")
    void shouldNotLeakResourcesOnRepeatedByteCalls() {
        var input = createByteInput();
        assertEquals(DataType.UINT8, input.getDataType());

        assertNoResourceGrowth(input, "UINT8");
    }

    @Test
    @DisplayName("The internal sub-manager must not survive the call")
    void shouldNotLeaveInternalSubManagersBehind() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);

        for (int i = 0; i < ITERATIONS; ++i) {
            try (var output = forward(input)) {
                assertNotNull(output);
            }
        }

        var dump = getDebugDump(root).orElseThrow(
                () -> new AssertionError("root manager should be a BaseNDManager"));
        var leftovers = dump.findMatching(
                node -> node.name().startsWith(INTERNAL_SUB_MANAGER_PREFIX));
        assertTrue(leftovers.isEmpty(),
                "safeForwardSingleFloat must close its internal sub-manager, but found "
                        + leftovers.size() + " leftover node(s): " + leftovers);
    }

    @Test
    @DisplayName("The output must be owned by the manager passed by the caller")
    void shouldReturnTheOutputOwnedByTheCallerManager() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);

        try (var output = forward(input)) {
            var array = output.singletonOrThrow();
            assertEquals(netManager, array.getManager(),
                    "the output must be attached to the caller manager");
            assertFalse(array.isReleased(), "the output must be alive when returned");
            assertEquals(OUTPUT_SHAPE, array.getShape());
            assertEquals(DataType.FLOAT32, array.getDataType());
        }
    }

    @Test
    @DisplayName("The caller input must survive the call and go back to its manager")
    void shouldNotReleaseTheCallerInput() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);
        float expected = input.getFloat(0, 0);

        try (var output = forward(input)) {
            assertNotNull(output);
        }

        assertFalse(input.isReleased(),
                "safeForwardSingleFloat must not release the input owned by the caller");
        assertEquals(netManager, input.getManager(),
                "the input must be returned to the manager that owns it");
        assertEquals(expected, input.getFloat(0, 0), 1e-6f,
                "the input content must not be modified by the forward pass");
    }

    @Test
    @DisplayName("A FLOAT32 input must reach the network without being converted")
    void shouldForwardFloat32InputWithoutConversion() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);

        capture(() -> {
            try (var output = forward(input)) {
                assertNotNull(output);
            }
        });

        var captured = capturedInput.get();
        assertNotNull(captured, "the network should have received an input");
        assertSame(input, captured,
                "a FLOAT32 input must be forwarded as-is, without an intermediate copy");
        assertFalse(captured.isReleased(),
                "the caller's input must be alive once the internal sub-manager is closed");
    }

    @Test
    @DisplayName("The internal FLOAT32 intermediate must be released after the call")
    void shouldReleaseTheInternalIntermediateOfTheBytePath() {
        var input = createByteInput();

        capture(() -> {
            try (var output = forward(input)) {
                assertNotNull(output);
            }
        });

        var captured = capturedInput.get();
        assertNotNull(captured, "the network should have received an input");
        assertNotSame(input, captured,
                "a byte input must be converted into a new FLOAT32 array");
        assertEquals(DataType.FLOAT32, capturedDataType.get(),
                "the array consumed by the network must already be FLOAT32");

        assertTrue(captured.isReleased(),
                "the internal intermediate must be closed when the internal sub-manager is closed, "
                        + "otherwise its native memory is leaked");
    }

    @Test
    @DisplayName("Repeated calls must not retain Java heap")
    void shouldNotRetainJavaHeap() {
        var input = netManager.randomUniform(0f, 1f, INPUT_SHAPE, DataType.FLOAT32);
        var runtime = Runtime.getRuntime();

        // Warm up so that lazy allocations are not accounted as growth.
        for (int i = 0; i < WARMUP; ++i) {
            try (var output = forward(input)) {
                assertNotNull(output);
            }
        }
        System.gc();

        long before = runtime.totalMemory() - runtime.freeMemory();
        for (int i = 0; i < ITERATIONS * 10; ++i) {
            try (var output = forward(input)) {
                assertNotNull(output);
            }
        }
        System.gc();

        long growth = (runtime.totalMemory() - runtime.freeMemory()) - before;
        assertTrue(growth < MAX_HEAP_GROWTH_BYTES,
                "Java heap grew by " + growth + " bytes over " + (ITERATIONS * 10) + " calls");
    }

    /**
     * Creates a {@code UINT8} input, mimicking the raw frame buffer produced by an
     * Atari environment. {@code NDManager.create(byte[], Shape)} would produce an
     * {@code INT8} array, so the data type is stated explicitly.
     */
    private NDArray createByteInput() {
        return netManager.create(ByteBuffer.wrap(BYTE_INPUT), INPUT_SHAPE, DataType.UINT8);
    }

    /**
     * Runs the given action while the network captures the array it consumes.
     */
    private void capture(Runnable action) {
        capturedInput.set(null);
        capturedDataType.set(null);
        recording.set(true);
        try {
            action.run();
        } finally {
            recording.set(false);
        }
    }

    /**
     * Asserts that the manager hierarchy does not grow across repeated calls.
     *
     * <p>The baseline is captured after a warm-up phase, because the very first call
     * also populates the {@link ParameterStore} and lazily initializes engine state.
     * Those allocations are one-off and would otherwise show up as a false positive.</p>
     *
     * @param input the input to forward
     * @param label a label used in the assertion messages
     */
    private void assertNoResourceGrowth(NDArray input, String label) {
        for (int i = 0; i < WARMUP; ++i) {
            try (var output = forward(input)) {
                assertFalse(output.singletonOrThrow().isReleased());
            }
        }

        int arraysBefore = managedArrayCount(root);
        int resourcesBefore = systemResourceCount(root);

        for (int i = 0; i < ITERATIONS; ++i) {
            try (var output = forward(input)) {
                assertFalse(output.singletonOrThrow().isReleased());
            }
        }

        int arraysAfter = managedArrayCount(root);
        int resourcesAfter = systemResourceCount(root);

        assertEquals(arraysBefore, arraysAfter,
                () -> "safeForwardSingleFloat leaked NDArrays on the " + label
                        + " path: before=" + arraysBefore + " after=" + arraysAfter
                        + " delta=" + (arraysAfter - arraysBefore));
        assertEquals(resourcesBefore, resourcesAfter,
                () -> "safeForwardSingleFloat leaked resources on the " + label
                        + " path: before=" + resourcesBefore + " after=" + resourcesAfter
                        + " delta=" + (resourcesAfter - resourcesBefore));
    }

    /**
     * Runs a forward pass through the method under test.
     *
     * <p>The {@link NDList} wrapping the input is intentionally not closed: it holds
     * the caller input, and closing it would release the caller's buffer. The method
     * under test never takes ownership of it.</p>
     */
    private NDList forward(NDArray input) {
        return safeForwardSingleFloat(
                netManager, net, parameterStore, new NDList(input), true);
    }
}
