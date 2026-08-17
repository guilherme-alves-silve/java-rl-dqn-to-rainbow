package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class IReplayBufferMemoryLeakTest {

    private static final int BATCH_SIZE = 32;
    private static final Shape STATE_SHAPE = new Shape(3, 3);

    private static NDManager testManager;
    private static NDManager systemManager;

    @BeforeAll
    static void setUp() {
        systemManager = NDManager.newBaseManager();
        testManager = systemManager.newSubManager();
    }

    @AfterAll
    static void shutdown() {
        testManager.close();
    }

    /**
     * Bigger batch to make sure the leak, if any, scales with the number of intermediate
     * operations (one {@code expandDims} per experience). This test will fail much more
     * loudly than the small one if the fix regresses.
     */
    @Test
    void newAttachedListShouldNotLeakOnLargeBatch() {
        int largeBatch = 512;
        var batch = createBatch(largeBatch);

        for (int i = 0; i < 5; ++i) {
            var out = invokeNewAttachedList(batch, i);
            assertNotNull(out);
        }

        int afterWarmup = managedArrayCount(systemManager);

        for (int i = 0; i < 20; ++i) {
            var out = invokeNewAttachedList(batch, i + 100);
            assertNotNull(out);
        }

        int afterSteps = managedArrayCount(systemManager);
        assertEquals(afterWarmup, afterSteps,
                "newAttachedList on a " + largeBatch + "-size batch must not leak. "
                        + "before=" + afterWarmup + " after=" + afterSteps
                        + " delta=" + (afterSteps - afterWarmup));
    }

    /**
     * The "shape" assertion: each call must produce an output with the right batch
     * dimension and the right per-element shape. This guards against a fix that closes
     * the leak but accidentally drops the batch dim or changes the dtype/device.
     */
    @Test
    void newAttachedListMustPreserveBatchShape() {
        var batch = createBatch(BATCH_SIZE);
        try (var callerSub = subMgr(testManager, "shape-check")) {
            var bridge = newBridge();
            var out = bridge.newAttachedList(callerSub, batch, e -> e.state().duplicate());
            var shape = out.getShape().getShape();
            assertEquals(BATCH_SIZE, shape[0],
                    "First axis must be the batch size");
            assertEquals(STATE_SHAPE.getShape().length + 1, shape.length,
                    "Output rank must be per-experience rank + 1 (batch axis)");
            for (int i = 0; i < STATE_SHAPE.getShape().length; ++i) {
                assertEquals(STATE_SHAPE.getShape()[i], shape[i + 1],
                        "Axis " + (i + 1) + " must match the per-experience shape");
            }
        }
    }

    /**
     * Closure contract: the returned array must belong to the caller-supplied subManager
     * (i.e. its lifetime is fully controlled by closing that sub). If a fix accidentally
     * attaches the result to a different manager, this test will fail on
     * {@code assertEquals(callerSub, out.getManager())}.
     */
    @Test
    void returnedConcatMustBelongToTheCallersSubManager() {
        var batch = createBatch(BATCH_SIZE);
        try (var callerSub = subMgr(testManager, "ownership")) {
            IReplayBuffer bridge = newBridge();
            var out = bridge.newAttachedList(callerSub, batch, e -> e.state().duplicate());
            assertEquals(callerSub, out.getManager(),
                    "Returned array must be attached to the caller-supplied subManager");
        }
    }

    /**
     * Calls {@link IReplayBuffer#newAttachedList} through a throwaway stub so we exercise
     * the default method without spinning up a full {@link ExperienceReplayBuffer}. The
     * returned array lives under {@code callerSub} and is closed by the try-with-resources
     * when this method returns.
     */
    private NDArray invokeNewAttachedList(Experience[] batch, int callId) {
        try (var callerSub = subMgr(testManager, "caller-" + callId)) {
            return newBridge().newAttachedList(callerSub, batch, e -> e.state().duplicate());
        }
    }

    /**
     * Minimal {@link IReplayBuffer} that exists only to expose the
     * {@link IReplayBuffer#newAttachedList default method}. None of the buffer-lifecycle
     * methods are exercised by these tests.
     */
    private static IReplayBuffer newBridge() {
        return new IReplayBuffer() {
            @Override public void store(Experience exp) { /* ignored */ }
            @Override public boolean enough(int batchSize) { return true; }
            @Override public IVecExperience sample(int batchSize) { return null; }
            @Override public int capacity() { return 0; }
            @Override public int size() { return 0; }
            @Override public boolean isOpen() { return true; }
            @Override public void close() { /* ignored */ }
        };
    }

    private Experience[] createBatch(int n) {
        var batch = new Experience[n];
        for (int i = 0; i < n; ++i) {
            var state = testManager.randomUniform(0, 10, STATE_SHAPE);
            var action = mock(ActionSpaceType.ActionResult.class);
            when(action.valueAs(Long.class)).thenReturn((long) (i));
            var nextState = testManager.randomNormal(STATE_SHAPE);
            batch[i] = new Experience(state, action, i, nextState, false);
        }
        return batch;
    }
}
