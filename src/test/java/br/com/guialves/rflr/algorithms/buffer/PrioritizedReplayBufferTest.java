package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.fixture.ExperienceFixture;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

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
        float minPriority = 0.000001f;
        @Cleanup var replayBuffer = new PrioritizedReplayBuffer(expectedCapacity, manager);
        for (int i = 0; i < 1; ++i) {
            float priority = (i * 0.1f) + minPriority;
            var exp = createRandomExperience(i, priority);
            if (i < expectedCapacity) assertEquals(i, replayBuffer.pos());
            else assertEquals(i % expectedCapacity, replayBuffer.pos());
            replayBuffer.store(exp);
            assertTrue(replayBuffer.size() <= expectedCapacity);
            assertDoesNotThrow(() -> replayBuffer.sample(1).close());
        }
    }

    private PrioritizedExperience createRandomExperience(int i, float priority) {
        return ExperienceFixture.createRandomPriorityExperience(manager, i, priority);
    }
}
