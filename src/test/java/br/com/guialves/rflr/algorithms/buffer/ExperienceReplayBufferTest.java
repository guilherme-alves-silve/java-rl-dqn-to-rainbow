package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.ThreadLocalRandom;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static br.com.guialves.rflr.fixture.ExperienceFixture.BATCH_1_SHAPE;
import static java.util.stream.IntStream.range;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ExperienceReplayBufferTest {

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
        var replayBuffer = new ExperienceReplayBuffer(expectedCapacity, manager);
        assertTrue(replayBuffer.isOpen());
        for (int i = 0; i < 1000; ++i) {
            var exp = createRandomExperience(i);
            if (i < expectedCapacity) assertEquals(i, replayBuffer.pos());
            else assertEquals(i % expectedCapacity, replayBuffer.pos());
            replayBuffer.store(exp);
            assertTrue(replayBuffer.size() <= expectedCapacity);
            assertDoesNotThrow(() -> replayBuffer.sample(1).close());
        }
        replayBuffer.close();
        assertFalse(replayBuffer.isOpen());
    }

    @Test
    void shouldSampleCorrectly() {

        int size = 10;
        int batchSize = 5;
        var expectedShape = new Shape(batchSize, 3, 3);

        var expectedDoneActions = manager.create(
                new long[] {0L, 2L, 4L, 6L, 8L}).reshape(BATCH_1_SHAPE);
        var expectedDone = manager.create(
                new float[] {1f, 1f, 1f, 1f, 1f}).reshape(BATCH_1_SHAPE);

        var expectedNotDoneActions = manager.create(
                new long[] {1L, 3L, 5L, 7L, 9L}).reshape(BATCH_1_SHAPE);
        var expectedNotDone = manager.create(
                new float[] {0f, 0f, 0f, 0f, 0f}).reshape(BATCH_1_SHAPE);

        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            @Cleanup var replayBuffer = new ExperienceReplayBuffer(size, manager);
            range(0, size).forEach(i -> replayBuffer.store(createRandomExperience(i)));

            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(anyInt(), anyInt()))
                    .thenReturn(0, 2, 4, 6, 8);

            assertTrue(replayBuffer.enough(batchSize));

            var vecExpDone = replayBuffer.sample(batchSize);
            assertEquals(size, replayBuffer.capacity());
            assertEquals(batchSize, vecExpDone.states().size(0));
            assertEquals(expectedShape, vecExpDone.states().getShape());
            assertTrue(vecExpDone.actions().contentEquals(expectedDoneActions));
            assertTrue(vecExpDone.dones().contentEquals(expectedDone));

            when(mockRandom.nextInt(anyInt(), anyInt()))
                    .thenReturn(1, 3, 5, 7, 9);

            var vecExpNotDone = replayBuffer.sample(batchSize);
            assertEquals(batchSize, vecExpNotDone.states().size(0));
            assertEquals(expectedShape, vecExpNotDone.states().getShape());
            assertTrue(vecExpNotDone.actions().contentEquals(expectedNotDoneActions));
            assertTrue(vecExpNotDone.dones().contentEquals(expectedNotDone));
        }
    }

    @ParameterizedTest(name = "[{index}] size={arguments}")
    @ValueSource(ints = {50, 100, 500, 1000, 5000})
    void shouldNotHaveMemoryLeak(int size) {
        int beforeAll = managedArrayCount(manager);
        int batchSize = 25;
        // this 2 plus objects are the state and nextState returned to the experience buffer
        int retToReplayBuffer = 2;
        var replayBuffer = new ExperienceReplayBuffer(size, manager);
        range(0, size).forEach(i -> replayBuffer.store(createRandomExperience(i)));

        int before = managedArrayCount(manager);
        try (var samples = replayBuffer.sample(batchSize)) {
            assertFalse(samples.states().isReleased());
            assertFalse(samples.actions().isReleased());
            assertFalse(samples.rewards().isReleased());
            assertFalse(samples.nextStates().isReleased());
            assertFalse(samples.dones().isReleased());
            assertTrue(managedArrayCount(manager) > before);
            assertNotEquals(samples.states().getManager(), manager);
            assertNotEquals(samples.actions().getManager(), manager);
            assertNotEquals(samples.rewards().getManager(), manager);
            assertNotEquals(samples.nextStates().getManager(), manager);
            assertNotEquals(samples.dones().getManager(), manager);
        }

        int after = managedArrayCount(manager);
        assertEquals(before + retToReplayBuffer, after);

        replayBuffer.close();
        assertEquals(beforeAll, managedArrayCount(manager));
    }

    private Experience createRandomExperience(int i) {
        return ExperienceFixture.createRandomExperience(manager, i);
    }
}
