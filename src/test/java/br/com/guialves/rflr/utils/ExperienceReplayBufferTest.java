package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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
    void should() {
        int size = 10;
        var replayBuffer = new ExperienceReplayBuffer(10, manager);

        for (int i = 0; i < size; ++i) {
            /*
             * NDArray state
             * ActionResult action
             * double reward
             * NDArray nextState
             * boolean done
             */
            var state = manager.randomUniform(0, 10, new Shape(3, 3));
            var action = mock(ActionSpaceType.ActionResult.class);
            int actionVal = i % 2 == 0? 1 : 0;
            //when(action.valueAs(Integer.class)).thenReturn()
            //var nextState = manager.randomNormal(new Shape(3, 3));
            //replayBuffer.store(new Experience());
        }

        //replayBuffer.store();
    }
}