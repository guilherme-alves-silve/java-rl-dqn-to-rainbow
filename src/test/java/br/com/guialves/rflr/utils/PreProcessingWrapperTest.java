package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.Gym;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

@Slf4j
class PreProcessingWrapperTest {

    private NDManager manager;
    private IEnv env;
    private PreProcessingWrapper wrapper;

    private static final int SKIP_FRAMES = 4;
    private static final int RESIZE_SIZE = 84;
    private static final int CONCATENATE_FRAMES = 4;
    private static final String ENV_NAME = "PongNoFrameskip-v4";

    @BeforeEach
    void setUp() {
        manager = NDManager.newBaseManager();
        env = Gym.builder()
                .importLib("ale_py")
                .ndManager(manager)
                .envName(ENV_NAME)
                .build();
        wrapper = new PreProcessingWrapper(env, SKIP_FRAMES, RESIZE_SIZE, CONCATENATE_FRAMES);
        assertThat(env.observationSpaceStr()).contains("Box");
    }

    @AfterEach
    @SneakyThrows
    void tearDown() {
        env.close();
        manager.close();
    }

    @Test
    void shouldReturnConcatenatedFrames() {
        var result = wrapper.reset();
        var state = result.getKey();
        var info = result.getValue();

        assertNotNull(state);
        assertNotNull(info);

        var expectedShape = new Shape(CONCATENATE_FRAMES, RESIZE_SIZE, RESIZE_SIZE);
        assertEquals(expectedShape, state.getShape());

        try (var firstFrame = state.get(0)) {
            for (int i = 1; i < CONCATENATE_FRAMES; i++) {
                try (var frame = state.get(i)) {
                    assertEquals(firstFrame, frame);
                }
            }
        }

        log.info("Reset state shape: {}, mean value: {}",
                state.getShape(), state.toType(DataType.FLOAT32, true).mean().getFloat());

        state.close();
    }

    @Test
    void shouldProcessSingleStep() {
        var resetResult = wrapper.reset();
        var initialState = resetResult.getKey();
        var action = env.actionSpaceSample();

        var stepResult = wrapper.step(action);
        var nextState = stepResult.state();

        assertNotNull(stepResult);
        assertNotNull(nextState);

        var expectedShape = new Shape(CONCATENATE_FRAMES, RESIZE_SIZE, RESIZE_SIZE);
        assertEquals(expectedShape, nextState.getShape());

        assertNotEquals(initialState, nextState);

        assertTrue(stepResult.reward() >= 0 && stepResult.reward() <= 1);

        assertNotNull(stepResult.info());

        log.info("Step reward: {}, done: {}, term: {}, trunc: {}",
                stepResult.reward(), stepResult.done(),
                stepResult.term(), stepResult.trunc());

        initialState.close();
        nextState.close();
    }

    @Test
    void shouldAccumulateRewardsFromSkipFrames() {
        wrapper.reset();
        var action = env.actionSpaceSample();

        try(var stepResult = wrapper.step(action)) {
            double reward = stepResult.reward();
            assertTrue(reward >= 0 && reward <= SKIP_FRAMES);
            log.info("Accumulated reward after {} skip frames: {}", SKIP_FRAMES, reward);
        }
    }

    @Test
    void shouldHandleEpisodeTermination() {
        wrapper.reset();
        var action = env.actionSpaceSample();

        int steps = 0;
        int maxSteps = 10_000;
        EnvStepResult stepResult = null;
        while (steps < maxSteps) {
            stepResult = wrapper.step(action);
            steps++;

            if (stepResult.done()) break;
            else stepResult.close();

            action = env.actionSpaceSample();
        }

        assertNotNull(stepResult);
        assertTrue(stepResult.done());
        assertTrue(steps > 0);

        log.info("Episode terminated after {} steps", steps);

        stepResult.close();
    }

    @Test
    void shouldMaintainConsistentStateShape() {
        wrapper.reset();
        var action = env.actionSpaceSample();
        List<Shape> shapes = new ArrayList<>();

        EnvStepResult stepResult = null;
        for (int i = 0; i < 10; i++) {
            stepResult = wrapper.step(action);
            shapes.add(stepResult.state().getShape());
            action = env.actionSpaceSample();

            if (stepResult.done()) break;
            else stepResult.close();
        }

        var expectedShape = new Shape(CONCATENATE_FRAMES, RESIZE_SIZE, RESIZE_SIZE);
        for (Shape shape : shapes) {
            assertEquals(expectedShape, shape);
        }

        stepResult.close();
    }

    @Test
    void shouldResetAfterDoneWorkCorrectly() {
        wrapper.reset();
        var action = env.actionSpaceSample();

        int maxSteps = 10_000;
        EnvStepResult stepResult = null;
        int steps1 = 0;

        while (steps1 < maxSteps) {
            stepResult = wrapper.step(action);
            steps1++;

            if (stepResult.done()) break;
            else stepResult.close();

            action = env.actionSpaceSample();
        }

        assertNotNull(stepResult);
        assertTrue(stepResult.done());

        var resetResult = wrapper.reset();
        var newState = resetResult.getKey();

        assertNotNull(newState);
        assertEquals(new Shape(CONCATENATE_FRAMES, RESIZE_SIZE, RESIZE_SIZE), newState.getShape());

        log.info("Episode 1 steps: {}, Episode 2 initial state shape: {}", steps1, newState.getShape());

        newState.close();
        stepResult.close();
    }

    @Test
    void shouldStepWorkWithDifferentSkipValues() {
        int[] skipValues = {1, 2, 4, 8};
        var expectedShape = new Shape(CONCATENATE_FRAMES, RESIZE_SIZE, RESIZE_SIZE);

        for (int skip : skipValues) {
            try (var testManager = NDManager.newBaseManager();
                 var testEnv = Gym.make(ENV_NAME, testManager)) {

                var testWrapper = new PreProcessingWrapper(testEnv, skip, RESIZE_SIZE, CONCATENATE_FRAMES);
                testWrapper.reset();
                var action = testEnv.actionSpaceSample();

                var stepResult = testWrapper.step(action);
                try (var state = stepResult.state()) {

                    assertEquals(expectedShape, state.getShape());

                    double reward = stepResult.reward();
                    assertTrue(reward <= skip + 0.1);

                    log.info("Skip={}, reward={}", skip, reward);
                }
            }
        }
    }

    @Test
    void shouldStepWorkWithDifferentConcatenateValues() {
        int[] concatValues = {1, 2, 4};
        for (int concat : concatValues) {
            var expectedSize = new Shape(concat, RESIZE_SIZE, RESIZE_SIZE);
            try (var testManager = NDManager.newBaseManager();
                 var testEnv = Gym.make(ENV_NAME, testManager)) {

                var testWrapper = new PreProcessingWrapper(testEnv, SKIP_FRAMES, RESIZE_SIZE, concat);
                testWrapper.reset();
                var action = testEnv.actionSpaceSample();

                try (var stepResult = testWrapper.step(action)) {
                    var state = stepResult.state();
                    assertEquals(expectedSize, state.getShape());
                    log.info("Concatenate={}, state shape={}", concat, state.getShape());
                }
            }
        }
    }

    @Test
    void shouldMemoryManagementNotLeak() {
        var runtime = Runtime.getRuntime();

        System.gc();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        int iterations = 100;
        var states = new ArrayList<NDArray>();

        for (int i = 0; i < iterations; i++) {
            wrapper.reset();
            var action = env.actionSpaceSample();

            for (int j = 0; j < 10; j++) {
                var stepResult = wrapper.step(action);

                if (j % 3 == 0) states.add(stepResult.state().duplicate());
                else stepResult.close();

                if (stepResult.done()) break;
                action = env.actionSpaceSample();
            }
        }

        states.stream()
                .filter(state -> !state.isReleased())
                .forEach(NDArray::close);

        System.gc();
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();

        long memoryDiff = memoryAfter - memoryBefore;
        log.info("Memory difference after {} iterations: {} bytes", iterations, memoryDiff);

        assertTrue(memoryDiff < 50_000_000, "Memory leak detected: " + memoryDiff + " bytes");
    }

    @Test
    void shouldExecuteInViableTime() {
        wrapper.reset();
        var action = env.actionSpaceSample();

        int steps = 100;
        long startTime = System.nanoTime();

        for (int i = 0; i < steps; i++) {
            try (var stepResult = wrapper.step(action)) {
                if (stepResult.done()) {
                    wrapper.reset();
                }
                action = env.actionSpaceSample();
            }
        }

        long endTime = System.nanoTime();
        long durationMs = (endTime - startTime) / 1_000_000;
        double stepsPerSecond = steps / (durationMs / 1000.0);

        log.info("Processed {} steps in {} ms = {} steps/second",
                steps, durationMs, stepsPerSecond);

        assertTrue(stepsPerSecond > 10, "Performance too slow: " + stepsPerSecond + " steps/second");
    }

    @Test
    void shouldConvertToGrayscale() {
        var resetResult = wrapper.reset();
        var state = resetResult.getKey();

        assertEquals(DataType.UINT8, state.getDataType());

        try (var minVal = state.min();
             var maxVal = state.max()) {
            assertTrue(minVal.getByte() >= 0);
            assertTrue(maxVal.getByte() <= 255);
        }

        log.info("Grayscale state - min: {}, max: {}, mean: {}",
                state.min().getByte(),
                state.max().getByte(),
                state.toType(DataType.FLOAT32, true).mean().getFloat());

        state.close();
    }

    @Test
    void shouldWorkWithDifferentEnvironments() {
        String[] testEnvs = {"CartPole-v1", "MountainCar-v0"};

        for (String envName : testEnvs) {
            try (var testManager = NDManager.newBaseManager();
                 var testEnv = Gym.make(envName, testManager)) {

                var testWrapper = new PreProcessingWrapper(testEnv, SKIP_FRAMES, RESIZE_SIZE, CONCATENATE_FRAMES);

                var resetResult = testWrapper.reset();
                var state = resetResult.getKey();
                var action = testEnv.actionSpaceSample();

                var stepResult = testWrapper.step(action);
                var nextState = stepResult.state();

                assertNotNull(state);
                assertNotNull(nextState);
                assertFalse(nextState.isReleased());
                assertEquals(state.getShape(), nextState.getShape());

                log.info("Environment: {}, state shape: {}", envName, state.getShape());

                state.close();
                assertTrue(nextState.isReleased());
            } catch (Exception e) {
                log.warn("Could not test environment {}: {}", envName, e.getMessage());
            }
        }
    }
}
