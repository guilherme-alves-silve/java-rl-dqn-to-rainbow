package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLUtils.gpuCount;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The multienvironment buffer: one buffer per environment, one mini-batch per
 * environment, merged into a single batch (env-major order).
 */
class NEnvReplayBufferTest {

    private static final int NUMBER_OF_ENVS = 2;
    private static final int CAPACITY = 16;
    private static final int EXPERIENCES_PER_ENV = 8;
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

    private NEnvReplayBuffer newBuffer(NDManager manager) {
        return new NEnvReplayBuffer(CAPACITY, NUMBER_OF_ENVS, manager,
                capacity -> new ExperienceReplayBuffer(capacity, manager));
    }
}
