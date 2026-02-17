package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.gymnasium4j.wrappers.*;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
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

    @Test
    void should() {

    }

    @Test
    void shouldTestGeneratedWrappers() {
        var envId = "CarRacing-v3";
        var script = Gym.builder()
                .envName(envId)
                //.params(Gym.builderMap().put("domain_randomize", true)
                //        .put("continuous", true))
                .add(new DelayObservation(1),
                     new GrayscaleObservation(false),
                     new NormalizeObservation(),
                     new MaxAndSkipObservation(4),
                     new FrameStackObservation(4),
                     new ReshapeObservation(new int[] {1, 84, 84}),
                     new ResizeObservation(new int[] {50, 50, 1}))
                .generatePyEnvScript();

        assertThat(script).contains("import gymnasium as gym");
        assertThat(script).contains("from gymnasium.wrappers import DelayObservation, GrayscaleObservation");
        assertThat(script).contains("NormalizeObservation, MaxAndSkipObservation, FrameStackObservation");
        assertThat(script).contains("ReshapeObservation, ResizeObservation");
        //assertThat(script).containsPattern("env_[0-9a-f]{32} = gym\\.make\\('CarRacing-v3', render_mode='rgb_array', \\{'domain_randomize': True, 'continuous': True}\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = gym\\.make\\('CarRacing-v3', render_mode='rgb_array'\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = DelayObservation\\(env_[0-9a-f]{32}, delay=1\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = GrayscaleObservation\\(env_[0-9a-f]{32}, keep_dim=False\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = NormalizeObservation\\(env_[0-9a-f]{32}, epsilon=1\\.0E-8\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = MaxAndSkipObservation\\(env_[0-9a-f]{32}, skip=4\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = FrameStackObservation\\(env_[0-9a-f]{32}, stack_size=4\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = ReshapeObservation\\(env_[0-9a-f]{32}, shape=\\[1, 84, 84]\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = ResizeObservation\\(env_[0-9a-f]{32}, shape=\\[50, 50, 1]\\)");
    }
}
