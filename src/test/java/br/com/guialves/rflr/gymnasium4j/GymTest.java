package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.python.PythonRuntime;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static br.com.guialves.rflr.python.PythonRuntime.insideGil;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j
class GymTest {

    @Test
    void shouldRunEnv() {
        var envName = "CartPole-v1";
        try (var render = Mockito.spy(new EnvRenderWindow());
             var ndManager = Mockito.spy(NDManager.newBaseManager());
             var env = Gym.make(envName, ndManager)) {

            //log.info("action space: {}, state space: {}", env.actionSpaceSample(), env.observationSpaceStr());
            env.reset();
            for (int frame = 1; frame <= 500; ++frame) {
                var img = env.render();
                assertEquals(400, img.getHeight());
                assertEquals(600, img.getWidth());
                render.display(img);
                render.waitRender();
                var action = env.actionSpaceSampleDouble();
                //var result = env.step(action);
                var result = env.step(action);

                if (result.done()) {

                }
            }
        }
    }
}
