package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.fixture.ExperienceFixture.createRandomExperience;
import static org.junit.jupiter.api.Assertions.*;

class PrioritizedReplayBufferTest {

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
    void shouldNotOverflow() {
        var expectedCapacity = 10;
        float alpha = 0.2f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(expectedCapacity, alpha, manager, Device.cpu());
        for (int i = 0; i < 100; ++i) {
            var exp = createRandomExperience(manager, i);
            if (i < expectedCapacity) assertEquals(i, replayBuffer.pos());
            else assertEquals(i % expectedCapacity, replayBuffer.pos());
            replayBuffer.store(exp);
            assertTrue(replayBuffer.size() <= expectedCapacity);
            assertDoesNotThrow(() -> replayBuffer.sample(1).close());
        }
    }

    @Test
    void shouldStoreExperienceAndUpdatePriorities() {
        int capacity = 5;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        var exp = createRandomExperience(manager, 0);
        replayBuffer.store(exp);

        assertEquals(1, replayBuffer.size());
        assertEquals(1, replayBuffer.pos());
        assertEquals(1.0f, replayBuffer.sumSegmentTree.get(0));
    }

    @Test
    void shouldOverwriteOldestExperienceWhenBufferFull() {
        int capacity = 3;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        // Fill buffer
        for (int i = 0; i < capacity; i++) {
            var exp = createRandomExperience(manager, i);
            replayBuffer.store(exp);
        }

        assertEquals(capacity, replayBuffer.size());
        assertEquals(0, replayBuffer.pos());

        // Add one more experience to trigger overwrite
        var newExp = createRandomExperience(manager, 99);
        replayBuffer.store(newExp);

        assertEquals(capacity, replayBuffer.size());
        assertEquals(1, replayBuffer.pos());
        // The experience at index 0 should be overwritten
        assertNotEquals(0, replayBuffer.pos());
    }

    // Tests for sample() method
    @Test
    void shouldSampleBatchWhenEnoughExperiences() {
        int capacity = 10;
        int batchSize = 3;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        for (int i = 0; i < batchSize + 2; i++) {
            var exp = createRandomExperience(manager, i);
            replayBuffer.store(exp);
        }

        var sampled = replayBuffer.sample(batchSize);

        assertNotNull(sampled);
        assertEquals(batchSize, sampled.states().getShape().get(0));
        assertEquals(batchSize, sampled.actions().getShape().get(0));
        assertEquals(batchSize, sampled.rewards().getShape().get(0));
        assertEquals(batchSize, sampled.nextStates().getShape().get(0));
        assertEquals(batchSize, sampled.dones().getShape().get(0));
        assertEquals(batchSize, sampled.weights().getShape().get(0));
        assertEquals(batchSize, sampled.bufferIndexes().length);

        sampled.close();
    }

    @Test
    void shouldReturnNullWhenNotEnoughExperiences() {
        int capacity = 10;
        int batchSize = 5;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        for (int i = 0; i < 3; i++) {
            var exp = createRandomExperience(manager, i);
            replayBuffer.store(exp);
        }

        var sampled = replayBuffer.sample(batchSize);
        assertNull(sampled);
    }

    @Test
    void shouldUpdatePrioritiesCorrectly() {
        int capacity = 10;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        for (int i = 0; i < 5; i++) {
            var exp = createRandomExperience(manager, i);
            replayBuffer.store(exp);
        }

        int[] bufferIndexes = {0, 1, 2};
        float[] priorities = {2.0f, 3.0f, 4.0f};
        try (var prioritiesArray = manager.create(priorities)) {
            replayBuffer.updatePriorities(bufferIndexes, prioritiesArray);
        }

        float expectedPriority0 = (float) Math.pow(2.0f, alpha);
        float expectedPriority1 = (float) Math.pow(3.0f, alpha);
        float expectedPriority2 = (float) Math.pow(4.0f, alpha);

        assertEquals(expectedPriority0, replayBuffer.sumSegmentTree.get(0), 0.001);
        assertEquals(expectedPriority1, replayBuffer.sumSegmentTree.get(1), 0.001);
        assertEquals(expectedPriority2, replayBuffer.sumSegmentTree.get(2), 0.001);
    }

    @Test
    void shouldThrowExceptionWhenUpdatingPrioritiesWithInvalidIndices() {
        int capacity = 10;
        float alpha = 0.5f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(capacity, alpha, manager, Device.cpu());

        for (int i = 0; i < 5; i++) {
            var exp = createRandomExperience(manager, i);
            replayBuffer.store(exp);
        }

        int[] bufferIndexes = {0, 1, 2};
        float[] priorities = {2.0f, 3.0f};
        try (NDArray prioritiesArray = manager.create(priorities)) {
            assertThrows(IllegalArgumentException.class, () ->
                    replayBuffer.updatePriorities(bufferIndexes, prioritiesArray));
        }

        int[] invalidIndexes = {0, 1, 20};
        float[] validPriorities = {2.0f, 3.0f, 4.0f};
        try (var prioritiesArray = manager.create(validPriorities)) {
            assertThrows(ArrayIndexOutOfBoundsException.class, () ->
                    replayBuffer.updatePriorities(invalidIndexes, prioritiesArray));
        }

        int[] zeroPriorityIndexes = {0, 1, 2};
        float[] zeroPriorities = {2.0f, 0.0f, 4.0f};
        try (var prioritiesArray = manager.create(zeroPriorities)) {
            assertThrows(IllegalArgumentException.class, () ->
                    replayBuffer.updatePriorities(zeroPriorityIndexes, prioritiesArray));
        }
    }
}
