package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import lombok.Cleanup;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLUtils.gpuCount;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * The multienvironment buffer: one buffer per environment, one mini-batch per
 * environment, merged into a single batch (env-major order).
 */
class NEnvReplayBufferTest {

    private static final int NUMBER_OF_ENVS = 2;
    private static final int CAPACITY = 16;
    private static final int EXPERIENCES_PER_ENV = 8;
    private static final int N_STEP = 2;
    private static final float GAMMA = 0.9f;
    private static final int BATCH_SIZE = 4;

    @Test
    void shouldStoreTheExperienceInTheBufferOfItsEnv() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            for (int env = 0; env < NUMBER_OF_ENVS; ++env) {
                for (int i = 0; i < EXPERIENCES_PER_ENV; ++i) {
                    buffer.store(ExperienceFixture.createRandomExperience(manager, env, i + (env * 100)));
                }
            }

            assertEquals(NUMBER_OF_ENVS * EXPERIENCES_PER_ENV, buffer.size());
            assertEquals(CAPACITY, buffer.capacity());
            assertEquals(CAPACITY / NUMBER_OF_ENVS, buffer.capacityPerEnv());
            assertTrue(buffer.isOpen());

            buffer.close();
            assertFalse(buffer.isOpen());
        }
    }

    @Test
    void shouldNotAcceptInnerBufferToHaveDifferentCapacityPerEnv() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var exception =  assertThrows(IllegalArgumentException.class, () -> new NEnvReplayBuffer(CAPACITY, NUMBER_OF_ENVS, manager,
                    _ -> new ExperienceReplayBuffer(100, manager)));
            assertEquals(exception.getMessage(), "The replay buffer in env 0 must use the capacityPerEnv 8");
        }
    }

    @Test
    void shouldNotBeEnoughWhenOneEnvHasNoData() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            assertFalse(buffer.enough(BATCH_SIZE));

            for (int i = 0; i < EXPERIENCES_PER_ENV; ++i) {
                buffer.store(ExperienceFixture.createRandomExperience(manager, 0, i));
            }
            assertFalse(buffer.enough(BATCH_SIZE), "the env 1 has no data yet");
            assertNull(buffer.sample(BATCH_SIZE));

            for (int i = 0; i < EXPERIENCES_PER_ENV; ++i) {
                buffer.store(ExperienceFixture.createRandomExperience(manager, 1, i + 100));
            }
            assertTrue(buffer.enough(BATCH_SIZE));

            buffer.close();
        }
    }

    @Test
    void shouldMergeTheBatchOfEveryEnv() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            for (int env = 0; env < NUMBER_OF_ENVS; ++env) {
                for (int i = 0; i < EXPERIENCES_PER_ENV; ++i) {
                    buffer.store(ExperienceFixture.createRandomExperience(manager, env, i + (env * 100)));
                }
            }

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertEquals(BATCH_SIZE, samples.states().getShape().get(0));
                assertEquals(new Shape(BATCH_SIZE, 3, 3), samples.states().getShape());
                assertEquals(BATCH_SIZE, samples.nextStates().getShape().get(0));
                assertEquals(BATCH_SIZE, samples.actions().getShape().get(0));

                // env-major order: the first half comes from the env 0, the second half from the env 1
                var actions = samples.actions().toLongArray();
                assertEquals(BATCH_SIZE, actions.length);
                for (int i = 0; i < BATCH_SIZE / NUMBER_OF_ENVS; ++i) {
                    assertTrue(actions[i] < 100, "the sample " + i + " must come from the env 0");
                    assertTrue(actions[i + (BATCH_SIZE / NUMBER_OF_ENVS)] >= 100,
                            "the sample " + i + " must come from the env 1");
                }
            }

            buffer.close();
        }
    }

    @Test
    void shouldSampleOnGpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = newBuffer(manager);
            for (int env = 0; env < NUMBER_OF_ENVS; ++env) {
                for (int i = 0; i < EXPERIENCES_PER_ENV; ++i) {
                    buffer.store(ExperienceFixture.createRandomExperience(manager, env, i + (env * 100)));
                }
            }

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertEquals(Device.gpu(), samples.states().getDevice());
                assertEquals(Device.gpu(), samples.actions().getDevice());
                assertEquals(Device.gpu(), samples.rewards().getDevice());
                assertEquals(Device.gpu(), samples.nextStates().getDevice());
                assertEquals(Device.gpu(), samples.dones().getDevice());
            }

            buffer.close();
        }
    }

    @Test
    void shouldRejectAnInvalidEnvNumber() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            var exp = ExperienceFixture.createRandomExperience(manager, NUMBER_OF_ENVS, 0);
            assertThrows(IllegalArgumentException.class, () -> buffer.store(exp));

            buffer.close();
        }
    }

    @Test
    void shouldAcceptABatchSizeThatIsNotAMultipleOfTheNumberOfEnvs() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            @Cleanup var buffer = newBiggerBuffer(manager);
            fill(buffer, manager, 8);

            // 10 samples over 4 environments: 3 + 3 + 2 + 2
            try (var samples = buffer.sample(10)) {
                assertEquals(10, samples.states().getShape().get(0));
                assertEquals(10, samples.actions().toLongArray().length);

                var actions = samples.actions().toLongArray();
                // env-major order: the env 0 gives 3, the env 1 gives 3, the env 2 gives 2, the env 3 gives 2
                assertEquals(0, actions[0]);
                assertEquals(0, actions[1]);
                assertEquals(0, actions[2]);
                assertEquals(1, actions[3]);
                assertEquals(1, actions[4]);
                assertEquals(1, actions[5]);
                assertEquals(2, actions[6]);
                assertEquals(2, actions[7]);
                assertEquals(3, actions[8]);
                assertEquals(3, actions[9]);
            }
        }
    }

    @Test
    void shouldKeepEveryEnvInTheBatch() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            fill(buffer, manager, 8);

            // the batch has one sample of each environment, plus the remainder
            try (var samples = buffer.sample(NUMBER_OF_ENVS + 2)) {
                var actions = samples.actions().toLongArray();
                var envsInTheBatch = new java.util.HashSet<Integer>();
                for (long action : actions) {
                    envsInTheBatch.add((int) action);
                }
                assertEquals(NUMBER_OF_ENVS, envsInTheBatch.size(),
                        "every environment must be part of the batch");
            }

            buffer.close();
        }
    }

    @Test
    void shouldRejectABatchSizeSmallerThanTheNumberOfEnvs() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newBuffer(manager);
            fill(buffer, manager, 8);

            var exception = assertThrows(IllegalArgumentException.class, () -> buffer.sample(NUMBER_OF_ENVS - 1));
            assertTrue(exception.getMessage().contains("must be at least the number of environments"),
                    "the error must explain the constraint: " + exception.getMessage());

            buffer.close();
        }
    }

    @Test
    void shouldComputeTheNStepReturnPerEnv() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newNStepBuffer(manager);

            // the env 0 gets the rewards 1 and 2, the env 1 gets the rewards 10 and 20
            buffer.store(experience(manager, 0, 1.0));
            buffer.store(experience(manager, 1, 10.0));
            buffer.store(experience(manager, 0, 2.0));
            buffer.store(experience(manager, 1, 20.0));

            // the n-step window of each env is complete: 1 + gamma*2 and 10 + gamma*20
            assertEquals(NUMBER_OF_ENVS, buffer.size(), "one n-step transition per env");

            try (var samples = buffer.sample(2)) {
                var rewards = samples.rewards().toFloatArray();
                assertEquals(2, rewards.length);

                // env-major order: the env 0 comes first
                assertEquals(1.0 + GAMMA * 2.0, rewards[0], 1e-4,
                        "the n-step return of the env 0 must not include the reward of the env 1");
                assertEquals(10.0 + GAMMA * 20.0, rewards[1], 1e-4,
                        "the n-step return of the env 1 must not include the reward of the env 0");
            }

            buffer.close();
        }
    }

    @Test
    void shouldExposeTheNStepOfTheBuffersOfTheEnvs() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newNStepBuffer(manager);
            assertEquals(N_STEP, buffer.nStep());
            buffer.close();
        }
    }

    @Test
    void shouldNotBeEnoughUntilEveryEnvCompletesItsNStepWindow() {
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = newNStepBuffer(manager);
            assertFalse(buffer.enough(2));

            // the env 0 completes its window, the env 1 does not
            buffer.store(experience(manager, 0, 1.0));
            buffer.store(experience(manager, 0, 2.0));
            assertFalse(buffer.enough(2), "the env 1 has no n-step transition yet");

            buffer.store(experience(manager, 1, 10.0));
            buffer.store(experience(manager, 1, 20.0));
            assertTrue(buffer.enough(2));

            buffer.close();
        }
    }

    private void fill(NEnvReplayBuffer buffer, NDManager manager, int perEnv) {
        for (int env = 0; env < buffer.numberOfEnvs(); ++env) {
            for (int i = 0; i < perEnv; ++i) {
                buffer.store(experience(manager, env));
            }
        }
    }

    private Experience experience(NDManager manager, int envNumber, double reward) {
        @Cleanup var state = manager.randomUniform(0, 10, ExperienceFixture.STATE_SHAPE);
        @Cleanup var action = mock(ActionSpaceType.ActionResult.class);
        when(action.valueAs(Long.class)).thenReturn((long) envNumber);
        when(action.duplicate()).thenReturn(action);
        @Cleanup var nextState = manager.randomNormal(ExperienceFixture.STATE_SHAPE);
        return Experience.copy(envNumber, state, action, reward, nextState, false, false);
    }

    private Experience experience(NDManager manager, int envNumber) {
        return experience(manager, envNumber, 1.0);
    }

    private NEnvReplayBuffer newNStepBuffer(NDManager manager) {
        return new NEnvReplayBuffer(CAPACITY, NUMBER_OF_ENVS, manager,
                capacity -> new NStepExperienceReplayBuffer(capacity, N_STEP, GAMMA, manager));
    }

    private NEnvReplayBuffer newBuffer(NDManager manager) {
        return new NEnvReplayBuffer(CAPACITY, NUMBER_OF_ENVS, manager,
                capacity -> new ExperienceReplayBuffer(capacity, manager));
    }

    private NEnvReplayBuffer newBiggerBuffer(NDManager manager) {
        int numberOfEnvs = 4;
        int totalCapacity = 64;
        return new NEnvReplayBuffer(totalCapacity, numberOfEnvs, manager,
                capacity -> new ExperienceReplayBuffer(capacity, manager));
    }
}
