package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import lombok.Cleanup;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test for the native memory leak in the Java &lt;-&gt; Python bridge.
 *
 * <p><b>Root cause.</b> The JavaCPP binding declares {@code org.bytedeco.cpython.PyObject}
 * as {@code @Opaque} <b>without a deallocator</b>. Therefore
 * {@code Pointer.close()} (which delegates to {@code releaseReference()}) and
 * {@code retainReference()} are both no-ops for {@code PyObject}: they never call
 * {@code Py_DECREF} and never invalidate the pointer. Every
 * {@code @Cleanup}/try-with-resources over a {@code PyObject} silently leaks one
 * Python reference.</p>
 *
 * <p>The APIs that hand a <b>new</b> reference to Java are
 * {@code PyObject_GetAttrString}, {@code PyObject_CallObject},
 * {@code PyObject_Str}, {@code PyObject_Type}, {@code PyUnicode_AsUTF8String} and
 * {@code PyRun_StringFlags}. Each of them must be balanced by an explicit
 * {@code Py_DECREF}.</p>
 *
 * <p>Before the fix, {@code Env.step()} leaked the returned tuple plus the observation
 * array on every step (~28.9 KB/step, which is what caused the native OOM in
 * {@code AgentCNNDQNMain}), and {@code Env.reset()} leaked 3 Python blocks per call
 * (the {@code shape} attribute and the {@code dtype} attribute chain).</p>
 */
class PythonBridgeMemoryLeakTest {

    private static final int WARMUP_STEPS = 200;
    private static final int MEASURED_STEPS = 2000;

    /**
     * Maximum number of Python blocks that may be allocated per call.
     * <p>The leak allocates one block for the step tuple and three for the reset
     * metadata, so a budget of 0.5 separates a leaking bridge (~1.0 and ~3.0) from a
     * clean one (~0.001) with a wide margin on both sides.</p>
     */
    private static final double MAX_BLOCKS_PER_CALL = 0.5;

    /**
     * Maximum native (commit charge) growth per step, in bytes.
     * <p>Before the fix, {@code Env.step()} leaked the observation array on every
     * step: ~526 bytes/step on CartPole (a 4x float64 observation) and ~28.9 KB/step
     * on Atari (a 4x84x84 uint8 observation, which is what caused the native OOM in
     * {@code AgentCNNDQNMain}). A budget of 256 bytes/step therefore catches the
     * leak on any environment, while staying well above the measurement noise
     * (the observed growth with the fixed bridge is 0 bytes).</p>
     */
    private static final long MAX_NATIVE_BYTES_PER_STEP = 256L;

    @Test
    @DisplayName("step() must not leak Python references (tuple + observation)")
    void stepShouldNotLeakPythonReferences() {
        try (var env = Gym.make("CartPole-v1");
             var manager = NDManager.newBaseManager()) {

            @Cleanup var action = env.actionSpaceSample();
            env.reset(manager).close();

            for (int i = 0; i < WARMUP_STEPS; i++) {
                env.step(action, manager).close();
            }

            long before = allocatedPythonBlocks();

            for (int i = 0; i < MEASURED_STEPS; i++) {
                env.step(action, manager).close();
            }

            long after = allocatedPythonBlocks();
            long leaked = after - before;
            double perStep = leaked / (double) MEASURED_STEPS;

            System.out.printf("blocks python: %d -> %d (leak = %d, %.4f per step)%n",
                    before, after, leaked, perStep);

            assertTrue(leaked >= 0, "The number of allocated Python blocks cannot decrease");
            assertTrue(perStep < MAX_BLOCKS_PER_CALL,
                    "Python blocks leaked per step: " + perStep
                            + " (expected < " + MAX_BLOCKS_PER_CALL + ")");
        }
    }

    @Test
    @DisplayName("reset() must not leak Python references (shape + dtype)")
    void resetShouldNotLeakPythonReferences() {
        try (var env = Gym.make("CartPole-v1");
             var manager = NDManager.newBaseManager()) {

            for (int i = 0; i < WARMUP_STEPS; i++) {
                env.reset(manager).close();
            }

            long before = allocatedPythonBlocks();

            for (int i = 0; i < MEASURED_STEPS; i++) {
                env.reset(manager).close();
            }

            long after = allocatedPythonBlocks();
            long leaked = after - before;
            double perReset = leaked / (double) MEASURED_STEPS;

            System.out.printf("python blocks (reset): %d -> %d (leak = %d, %.4f por reset)%n",
                    before, after, leaked, perReset);

            assertTrue(perReset < MAX_BLOCKS_PER_CALL,
                    "Python blocks leaked per reset: " + perReset
                            + " (expected < " + MAX_BLOCKS_PER_CALL + ")");
        }
    }

    @Test
    @DisplayName("step() must not grow the native commit charge")
    void stepShouldNotLeakNativeMemory() {
        try (var env = Gym.make("CartPole-v1");
             var manager = NDManager.newBaseManager()) {

            @Cleanup var action = env.actionSpaceSample();

            // gym.make() applies the OrderEnforcing wrapper, which rejects step()
            // before the first reset().
            env.reset(manager).close();

            for (int i = 0; i < WARMUP_STEPS; i++) {
                env.step(action, manager).close();
            }

            long before = committedNativeBytes();

            for (int i = 0; i < MEASURED_STEPS; i++) {
                env.step(action, manager).close();
            }

            long after = committedNativeBytes();
            long growth = after - before;
            long perStep = growth / MEASURED_STEPS;

            System.out.printf("commit charge: %d -> %d (growth = %d bytes, %d por step)%n",
                    before, after, growth, perStep);

            assertTrue(perStep < MAX_NATIVE_BYTES_PER_STEP,
                    "Native memory leaked per step: " + perStep + " bytes"
                            + " (expected < " + MAX_NATIVE_BYTES_PER_STEP + " bytes)");
        }
    }

    /**
     * @return the commit charge of the process, in bytes. This is the metric that
     *         reflects native (malloc) allocations, which the Java heap does not
     *         see. It must be read through the {@code com.sun.management} interface:
     *         the JDK implementation class is package-private.
     */
    private static long committedNativeBytes() {
        var os = (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        return os.getCommittedVirtualMemorySize();
    }
}
