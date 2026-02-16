package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static br.com.guialves.rflr.python.PythonRuntime.insideGil;
import static org.junit.jupiter.api.Assertions.*;

@Slf4j
class GymTest {

    @Test
    void shouldRunEnv() {
        // https://gymnasium.farama.org/environments/classic_control/cart_pole/
        var envName = "CartPole-v1";
        try (var render = Mockito.spy(new EnvRenderWindow());
             var ndManager = Mockito.spy(NDManager.newBaseManager());
             var env = Gym.make(envName, ndManager)) {

            log.info("action space: {}, state space: {}", env.actionSpaceStr(), env.observationSpaceStr());

            assertTrue(env.actionSpaceStr().contains("Discrete(2)"));
            assertTrue(env.observationSpaceStr().contains("Box([-4.8"));

            env.reset();
            int frame;
            EnvStepResult stepResult = null;
            for (frame = 1; frame <= 100; ++frame) {
                var img = env.render();
                assertEquals(400, img.getHeight());
                assertEquals(600, img.getWidth());
                render.display(img);
                render.waitRender();
                var action = env.actionSpaceSample();
                stepResult = env.step(action);
                if (stepResult.term()) {
                    break;
                }

            }

            assertNotNull(stepResult);
            assertNotNull(stepResult.info());
            assertTrue(stepResult.done());
            assertTrue(stepResult.term());
            assertTrue(frame > 0);
        }
    }
}
