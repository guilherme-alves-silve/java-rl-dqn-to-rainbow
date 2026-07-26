package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.DISCRETE;
import static org.junit.jupiter.api.Assertions.*;

class NStepExperienceReplayBufferTest {

    private static final float DELTA = 1e-6f;
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
    void shouldRejectInvalidNStep() {
        assertThrows(IllegalArgumentException.class,
                () -> new NStepExperienceReplayBuffer(0, 0.99f, 10, manager));
        assertThrows(IllegalArgumentException.class,
                () -> new NStepExperienceReplayBuffer(3, 1.5f, 10, manager));
    }

    @Test
    void shouldDelegateEnough() {
        int nStep = 2;
        @Cleanup var buffer = new NStepExperienceReplayBuffer(nStep, 0.99f, 10, manager);
        assertFalse(buffer.enough(1));
        buffer.store(exp(manager, 0, 1.0));
        buffer.store(exp(manager, 1, 1.0));
        assertTrue(buffer.enough(1));
    }

    /**
     * Get the scenario below:
     * (done)ABC
     */
    @Test
    void shouldEmitFirstDoneIfItEndsEpisode() {
        int nStep = 3;
        float gamma = 0.99f;
        int capacity = 100;
        int expectedSize = 1;

        @Cleanup var buffer = new NStepExperienceReplayBuffer(nStep, gamma, capacity, manager);

        buffer.store(expDone(manager, 0, 1.0));
        buffer.store(exp(manager, 1, 1.0));
        buffer.store(exp(manager, 2, 1.0));

        @Cleanup var samples = buffer.sample(1);

        assertEquals(expectedSize, samples.dones().size(0));
        assertEquals(expectedSize, buffer.size());
    }

    /**
     * Get the scenario below:
     * ABC(done)
     * BC(done)
     * C(done)
     */
    @Test
    void shouldEmitAllUntilDone() {
        int nStep = 3;
        float gamma = 0.99f;
        int capacity = 100;
        int expectedSize = 3;

        @Cleanup var buffer = new NStepExperienceReplayBuffer(nStep, gamma, capacity, manager);

        buffer.store(exp(manager, 0, 1.0));
        buffer.store(exp(manager, 1, 0.8));
        buffer.store(expDone(manager, 2, 0.5));

        @Cleanup var samples = buffer.sample(3);

        assertEquals(expectedSize, samples.dones().size(0));
        assertEquals(expectedSize, buffer.size());
    }

    @Test
    void shouldEmitSlidingWindowAndFlushForLongEpisode() {
        int nStep = 3;
        float gamma = 1.0f;

        @Cleanup var buffer = new NStepExperienceReplayBuffer(nStep, gamma, 100, manager);

        buffer.store(exp(manager, 0, 1.0));
        buffer.store(exp(manager, 1, 2.0));
        buffer.store(exp(manager, 2, 3.0));
        buffer.store(exp(manager, 3, 4.0));
        buffer.store(expDone(manager, 4, 5.0));

        assertEquals(5, buffer.size());

        @Cleanup var samples = buffer.sample(5);

        var rewards = samples.rewards().toFloatArray();

        Arrays.sort(rewards);

        assertArrayEquals(
                new float[]{5.0f, 6.0f, 9.0f, 9.0f, 12.0f},
                rewards,
                DELTA
        );
    }

    @Test
    void shouldComputeDiscountedNStepReward() {
        int nStep = 3;
        float gamma = 0.5f;

        @Cleanup
        var buffer = new NStepExperienceReplayBuffer(nStep, gamma, 100, manager);

        buffer.store(exp(manager, 0, 1.0));
        buffer.store(exp(manager, 1, 2.0));
        buffer.store(expDone(manager, 2, 3.0));

        assertEquals(3, buffer.size());

        @Cleanup
        var samples = buffer.sample(3);

        var rewards = samples.rewards().toFloatArray();

        Arrays.sort(rewards);

        assertArrayEquals(
                new float[]{
                        2.75f,
                        3.0f,
                        3.5f
                },
                rewards,
                DELTA
        );
    }

    private Experience exp(NDManager mgr, int seed, double reward) {
        return buildExp(mgr, seed, reward, false);
    }

    private Experience expDone(NDManager mgr, int seed, double reward) {
        return buildExp(mgr, seed, reward, true);
    }

    private Experience buildExp(NDManager mgr, int seed, double reward, boolean done) {
        var state = mgr.ones(new Shape(2, 2));
        var nextState = mgr.zeros(new Shape(2, 2));
        var action = DISCRETE.get(seed);
        return new Experience(state, action, reward, nextState, done);
    }
}
