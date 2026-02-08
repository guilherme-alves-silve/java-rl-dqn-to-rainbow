package br.com.guialves.rflr.gymnasium4j;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.*;

class GymTest {

    @Test
    void shouldRunEnv() {
        var envName = "CartPole-v1";
        try (var render = Mockito.spy(new EnvRenderWindow());
             var ndManager = Mockito.spy(NDManager.newBaseManager());
             var env = Gym.make(envName, ndManager)) {

            //log.info("action space: {}, state space: {}", env.actionSpaceSample(), env.observationSpaceStr());

            while (!Thread.currentThread().isInterrupted()) {
                render.display(env.render());

                render.waitRender();
            }
        }
    }
}