package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class PreProcessingWrapperTest {

    private NDManager manager;
    private IEnv env;

    @BeforeEach
    void setup() {
        manager = NDManager.newBaseManager();
        env = mock(IEnv.class);

        // env.getManager() is called inside step() and reset() to obtain the
        // parent for state.attach(parent) before the sub-manager closes
        when(env.manager()).thenReturn(manager);

        // The real env.step() creates the state NDArray on the sub-manager it
        // receives. The mock must do the same: capture that sub-manager and
        // create the frame on it, so ownership is correct throughout the wrapper.
        when(env.step(any(), any(NDManager.class))).thenAnswer(inv -> {
            NDManager sub = inv.getArgument(1);
            NDArray frame = createRgbFrame(sub, 84, 84);
            return new EnvStepResult(1.0, false, false, new HashMap<>(), frame);
        });

        // env.reset() returns a frame attached to the base manager (before any
        // sub is created), matching how the real implementation behaves
        when(env.reset()).thenAnswer(inv -> {
            NDArray frame = createRgbFrame(manager, 84, 84);
            return new Pair<>(new HashMap<>(), frame);
        });
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    // Creates a realistic RGB frame [H, W, 3] UINT8 on the given manager,
    // matching the shape the real Gymnasium environment produces
    private NDArray createRgbFrame(NDManager mgr, int h, int w) {
        try (NDArray f32 = mgr.randomUniform(0f, 255f, new Shape(h, w, 3), DataType.FLOAT32)) {
            return f32.toType(DataType.UINT8, false);
        }
    }

    @Test
    void testResetProducesConcatenatedFrames() {
        var wrapper = new PreProcessingWrapper(env, 0, 42, 4);
        var result = wrapper.reset();
        var state = result.getKey();

        // concatenate=4 grayscale frames stacked: [4, 42, 42]
        assertEquals(new Shape(4, 42, 42), state.getShape());
        assertEquals(DataType.UINT8, state.getDataType());
    }

    @Test
    void testStepProducesConcatenatedFrames() {
        var wrapper = new PreProcessingWrapper(env, 0, 42, 4);
        var result = wrapper.step(mock(ActionSpaceType.ActionResult.class));
        var state = result.state();

        assertEquals(new Shape(4, 42, 42), state.getShape());
        assertEquals(DataType.UINT8, state.getDataType());
    }

    @Test
    void testStepSumsRewardsAcrossConcatenatedFrames() {
        // skip=0, concatenate=4, each step returns reward=1.0 → total=4.0
        var wrapper = new PreProcessingWrapper(env, 0, 42, 4);
        var result = wrapper.step(mock(ActionSpaceType.ActionResult.class));

        assertEquals(4.0, result.reward(), 1e-6);
        assertFalse(result.term());
        assertFalse(result.trunc());
    }

    @Test
    void testSkipLogic() {
        // skip=2 → skipFrames loops i < 2 → exactly 2 env.step() calls per
        // concatenated frame; concatenate=1 → total env.step() calls = 2
        var wrapper = new PreProcessingWrapper(env, 2, 42, 1);
        wrapper.step(mock(ActionSpaceType.ActionResult.class));

        verify(env, times(2)).step(any(), any(NDManager.class));
    }

    @Test
    void testTerminatesEarlyOnDone() {
        // Override: every step signals termination
        when(env.step(any(), any(NDManager.class))).thenAnswer(inv -> {
            NDManager sub = inv.getArgument(1);
            NDArray frame = createRgbFrame(sub, 84, 84);
            return new EnvStepResult(1.0, true, false, new HashMap<>(), frame);
        });

        // concatenate=4 but exits after first frame; remaining 3 are padded
        var wrapper = new PreProcessingWrapper(env, 0, 42, 4);
        var result = wrapper.step(mock(ActionSpaceType.ActionResult.class));

        // Only 1 reward collected before early exit
        assertEquals(1.0, result.reward(), 1e-6);
        assertTrue(result.term());
        assertFalse(result.trunc());

        // Shape must still be complete — padding fills the missing frames
        assertEquals(new Shape(4, 42, 42), result.state().getShape());

        // env.step() must have been called exactly once (skip=0, stopped at i=0)
        verify(env, times(1)).step(any(), any(NDManager.class));
    }
}
