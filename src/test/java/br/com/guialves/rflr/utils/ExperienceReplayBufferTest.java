package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static java.util.stream.IntStream.range;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;

class ExperienceReplayBufferTest {

    private static final Shape STATE_SHAPE = new Shape(3, 3);
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
        var replayBuffer = new ExperienceReplayBuffer(expectedCapacity, STATE_SHAPE, manager);
        for (int i = 0; i < 1000; ++i) {
            var exp = createRandomExperience(i);
            if (i < expectedCapacity) assertEquals(i, replayBuffer.pos());
            else assertEquals(i % expectedCapacity, replayBuffer.pos());
            replayBuffer.store(exp);
            assertTrue(replayBuffer.size() <= expectedCapacity);
        }
    }

    @Test
    void shouldSampleCorrectly() {

        int size = 10;
        int batchSize = 5;
        var expectedShape = new Shape(batchSize, 3, 3);

        var expectedDoneActions = manager.create(new int[] {0, 2, 4, 6, 8});
        var expectedDone = manager.create(new int[] {1, 1, 1, 1, 1});

        var expectedNotDoneActions = manager.create(new int[] {1, 3, 5, 7, 9});
        var expectedNotDone = manager.create(new int[] {0, 0, 0, 0, 0});

        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var replayBuffer = new ExperienceReplayBuffer(size, STATE_SHAPE, manager);
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

    private Experience createRandomExperience(int i) {
        var state = manager.randomUniform(0, 10, STATE_SHAPE);
        var action = mock(ActionSpaceType.ActionResult.class);
        when(action.valueAs(int.class)).thenReturn(i);
        var reward = -5 + (i + 1);
        var nextState = manager.randomNormal(STATE_SHAPE);
        boolean done = i % 2 == 0;
        return new Experience(state, action, reward, nextState, done);
    }
}
