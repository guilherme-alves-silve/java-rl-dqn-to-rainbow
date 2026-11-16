package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static br.com.guialves.rflr.djlutils.DJLUtils.gpuCount;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The mini-batch delivered by {@link IReplayBuffer#sample(int)} must live on the
 * training device (the GPU), even though the transitions are stored in the CPU.
 */
class ReplayBufferGpuDeviceTest {

    private static final int BUFFER_CAPACITY = 16;
    private static final int BATCH_SIZE = 4;
    private static final int N_EXPERIENCES = 8;

    @Test
    void shouldSampleOnGpuWithExperienceReplayBuffer() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = new ExperienceReplayBuffer(BUFFER_CAPACITY, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
            }
            buffer.close();
        }
    }

    @Test
    void shouldSampleOnGpuWithPrioritizedReplayBuffer() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = new PrioritizedReplayBuffer(BUFFER_CAPACITY, 0.5f, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
                assertEquals(Device.gpu(), samples.weights().getDevice());
            }
            buffer.close();
        }
    }

    @Test
    void shouldSampleOnGpuWithNStepPrioritizedReplayBuffer() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = new NStepPrioritizedReplayBuffer(BUFFER_CAPACITY, 3, 0.99f, 0.5f, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
                assertEquals(Device.gpu(), samples.weights().getDevice());
            }
            buffer.close();
        }
    }

    @Test
    void shouldSampleOnGpuWithNEnvReplayBuffer() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        int numberOfEnvs = 4;
        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = new NEnvReplayBuffer(BUFFER_CAPACITY, numberOfEnvs, manager,
                    capacity -> new ExperienceReplayBuffer(capacity, manager));
            for (int env = 0; env < numberOfEnvs; ++env) {
                for (int i = 0; i < N_EXPERIENCES; ++i) {
                    buffer.store(ExperienceFixture.createRandomExperience(manager, env, i));
                }
            }

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
            }
            buffer.close();
        }
    }

    @Test
    void shouldNotLeakWhenSamplingOnGpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        try (var manager = NDManager.newBaseManager(Device.gpu())) {
            var buffer = new ExperienceReplayBuffer(BUFFER_CAPACITY, manager);
            fill(buffer, manager);

            int before = managedArrayCount(manager);
            for (int i = 0; i < 10; ++i) {
                try (var samples = buffer.sample(BATCH_SIZE)) {
                    assertOnGpu(samples);
                }
            }
            assertEquals(before, managedArrayCount(manager), "the sampled batches must be released");

            buffer.close();
        }
    }

    @Test
    void shouldSampleTheWeightsOnGpuEvenWhenTheParentIsOnTheCpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        // The importance sampling weights must live in the same device as the rest of
        // the mini-batch, otherwise the batch is split across two devices.
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = new PrioritizedReplayBuffer(BUFFER_CAPACITY, 0.5f, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
                assertEquals(Device.gpu(), samples.weights().getDevice(), "the weights must be on the GPU");
            }
            buffer.close();
        }
    }

    @Test
    void shouldSampleOnGpuEvenWhenTheParentIsOnTheCpu() {
        assumeTrue(gpuCount() > 0, "GPU is not available");

        // Device policy: everything that is computed lives in the GPU, even when the
        // parent manager is on the CPU.
        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = new ExperienceReplayBuffer(BUFFER_CAPACITY, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertOnGpu(samples);
            }
            buffer.close();
        }
    }

    @Test
    void shouldSampleOnCpuWhenThereIsNoGpu() {
        assumeTrue(gpuCount() == 0, "GPU is available");

        try (var manager = NDManager.newBaseManager(Device.cpu())) {
            var buffer = new ExperienceReplayBuffer(BUFFER_CAPACITY, manager);
            fill(buffer, manager);

            try (var samples = buffer.sample(BATCH_SIZE)) {
                assertEquals(Device.cpu(), samples.states().getDevice());
                assertEquals(Device.cpu(), samples.actions().getDevice());
                assertEquals(Device.cpu(), samples.rewards().getDevice());
                assertEquals(Device.cpu(), samples.nextStates().getDevice());
                assertEquals(Device.cpu(), samples.dones().getDevice());
            }
            buffer.close();
        }
    }

    private void fill(IReplayBuffer buffer, NDManager manager) {
        for (int i = 0; i < N_EXPERIENCES; ++i) {
            buffer.store(ExperienceFixture.createRandomExperience(manager, i));
        }
    }

    private void assertOnGpu(IVecExperience samples) {
        assertEquals(Device.gpu(), samples.states().getDevice(), "states must be on the GPU");
        assertEquals(Device.gpu(), samples.actions().getDevice(), "actions must be on the GPU");
        assertEquals(Device.gpu(), samples.rewards().getDevice(), "rewards must be on the GPU");
        assertEquals(Device.gpu(), samples.nextStates().getDevice(), "nextStates must be on the GPU");
        assertEquals(Device.gpu(), samples.dones().getDevice(), "dones must be on the GPU");
    }
}
