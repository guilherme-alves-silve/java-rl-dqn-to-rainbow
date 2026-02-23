package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.gymnasium4j.wrappers.*;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

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
    void shouldTestDefaultGeneratedWrapper() {
        var envId = "CarRacing-v3";
        var script = Gym.builder()
                .envName(envId)
                .generatePyEnvScript();

        assertThat(script).contains("import gymnasium as gym");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = gym\\.make\\('CarRacing-v3', render_mode='rgb_array'\\)");
        assertThat(script).doesNotContain("ale_py");
        assertThat(script).doesNotContain("from gymnasium.wrappers import DelayObservation, GrayscaleObservation");
        assertThat(script).doesNotContain("NormalizeObservation, MaxAndSkipObservation, FrameStackObservation");
        assertThat(script).doesNotContain("ReshapeObservation, ResizeObservation");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = gym\\.make\\('CarRacing-v3', render_mode='rgb_array', domain_randomize=True, continuous=True\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = DelayObservation\\(env_[0-9a-f]{32}, delay=1\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = GrayscaleObservation\\(env_[0-9a-f]{32}, keep_dim=False\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = NormalizeObservation\\(env_[0-9a-f]{32}, epsilon=1\\.0E-8\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = MaxAndSkipObservation\\(env_[0-9a-f]{32}, skip=4\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = FrameStackObservation\\(env_[0-9a-f]{32}, stack_size=4\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = ReshapeObservation\\(env_[0-9a-f]{32}, shape=\\[1, 84, 84]\\)");
        assertThat(script).doesNotContainPattern("env_[0-9a-f]{32} = ResizeObservation\\(env_[0-9a-f]{32}, shape=\\[50, 50, 1]\\)");
    }

    @Test
    void shouldTestGeneratedWrappers() {
        var envId = "CarRacing-v3";
        var script = Gym.builder()
                .envName(envId)
                .importLib("ale_py")
                .params(Gym.builderMap()
                        .put("domain_randomize", true)
                        .put("continuous", true))
                .add(new DelayObservation(1),
                     new GrayscaleObservation(false),
                     new NormalizeObservation(),
                     new MaxAndSkipObservation(4),
                     new FrameStackObservation(4),
                     new ReshapeObservation(new int[] {1, 84, 84}),
                     new ResizeObservation(new int[] {50, 50, 1}))
                .generatePyEnvScript();

        assertThat(script).contains("import gymnasium as gym, ale_py");
        assertThat(script).contains("from gymnasium.wrappers import DelayObservation, GrayscaleObservation");
        assertThat(script).contains("NormalizeObservation, MaxAndSkipObservation, FrameStackObservation");
        assertThat(script).contains("ReshapeObservation, ResizeObservation");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = gym\\.make\\('CarRacing-v3', render_mode='rgb_array', domain_randomize=True, continuous=True\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = DelayObservation\\(env_[0-9a-f]{32}, delay=1\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = GrayscaleObservation\\(env_[0-9a-f]{32}, keep_dim=False\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = NormalizeObservation\\(env_[0-9a-f]{32}, epsilon=1\\.0E-8\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = MaxAndSkipObservation\\(env_[0-9a-f]{32}, skip=4\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = FrameStackObservation\\(env_[0-9a-f]{32}, stack_size=4\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = ReshapeObservation\\(env_[0-9a-f]{32}, shape=\\[1, 84, 84]\\)");
        assertThat(script).containsPattern("env_[0-9a-f]{32} = ResizeObservation\\(env_[0-9a-f]{32}, shape=\\[50, 50, 1]\\)");
    }

    @Nested
    @DisplayName("PyMap Tests")
    class PyMapTest {

        @Test
        void shouldCreateEmptyPyMap() {
            var pyMap = Gym.builderMap();

            assertTrue(pyMap.isEmpty());
            assertEquals(0, pyMap.size());
            assertEquals("", pyMap.toPyKwargs());
            assertEquals("{}", pyMap.toPyDict());
        }

        @Test
        void shouldAddBooleanParameters() {
            var pyMap = Gym.builderMap()
                    .put("domain_randomize", true)
                    .put("continuous", false);

            assertFalse(pyMap.isEmpty());
            assertEquals(2, pyMap.size());
            assertEquals("domain_randomize=True, continuous=False", pyMap.toPyKwargs());
            assertEquals("{'domain_randomize': True, 'continuous': False}", pyMap.toPyDict());
        }

        @Test
        void shouldAddIntegerParameters() {
            var pyMap = Gym.builderMap()
                    .put("max_episode_steps", 1000)
                    .put("seed", 42);

            assertEquals(2, pyMap.size());
            assertEquals("max_episode_steps=1000, seed=42", pyMap.toPyKwargs());
            assertEquals("{'max_episode_steps': 1000, 'seed': 42}", pyMap.toPyDict());
        }

        @Test
        void shouldAddDoubleParameters() {
            var pyMap = Gym.builderMap()
                    .put("gravity", -9.8)
                    .put("force_mag", 10.5);

            assertEquals(2, pyMap.size());
            assertEquals("gravity=-9.8, force_mag=10.5", pyMap.toPyKwargs());
            assertEquals("{'gravity': -9.8, 'force_mag': 10.5}", pyMap.toPyDict());
        }

        @Test
        void shouldAddRawStringParameters() {
            var pyMap = Gym.builderMap()
                    .put("render_mode", "'human'")
                    .put("apply_api_compatibility", "True");

            assertEquals(2, pyMap.size());
            assertEquals("render_mode='human', apply_api_compatibility=True", pyMap.toPyKwargs());
        }

        @Test
        void shouldAddQuotedStringParameters() {
            var pyMap = Gym.builderMap()
                    .putStr("render_mode", "human")
                    .putStr("observation_mode", "rgb");

            assertEquals(2, pyMap.size());
            assertEquals("render_mode='human', observation_mode='rgb'", pyMap.toPyKwargs());
            assertEquals("{'render_mode': 'human', 'observation_mode': 'rgb'}", pyMap.toPyDict());
        }

        @Test
        void shouldAddMixedTypeParameters() {
            var pyMap = Gym.builderMap()
                    .put("continuous", true)
                    .put("max_episode_steps", 1000)
                    .put("gravity", -9.8)
                    .putStr("render_mode", "human");

            assertEquals(4, pyMap.size());
            assertThat(pyMap.toPyKwargs())
                    .contains("continuous=True")
                    .contains("max_episode_steps=1000")
                    .contains("gravity=-9.8")
                    .contains("render_mode='human'");
        }

        @Test
        void shouldOverwriteExistingParameter() {
            var pyMap = Gym.builderMap()
                    .put("seed", 42)
                    .put("seed", 100);

            assertEquals(1, pyMap.size());
            assertEquals("seed=100", pyMap.toPyKwargs());
        }

        @Test
        void shouldPreserveInsertionOrder() {
            var pyMap = Gym.builderMap()
                    .put("first", 1)
                    .put("second", 2)
                    .put("third", 3);

            assertEquals("first=1, second=2, third=3", pyMap.toPyKwargs());
        }

        @Test
        void shouldConvertToStringUsingToPyKwargs() {
            var pyMap = Gym.builderMap()
                    .put("domain_randomize", true)
                    .put("continuous", false);

            assertEquals("domain_randomize=True, continuous=False", pyMap.toString());
        }

        @Test
        void shouldHandleSingleParameter() {
            var pyMap = Gym.builderMap()
                    .put("max_episode_steps", 500);

            assertEquals("max_episode_steps=500", pyMap.toPyKwargs());
            assertEquals("{'max_episode_steps': 500}", pyMap.toPyDict());
        }

        @Test
        void shouldHandleNegativeNumbers() {
            var pyMap = Gym.builderMap()
                    .put("min_value", -100)
                    .put("gravity", -9.8);

            assertThat(pyMap.toPyKwargs())
                    .contains("min_value=-100")
                    .contains("gravity=-9.8");
        }

        @Test
        void shouldHandleZeroValues() {
            var pyMap = Gym.builderMap()
                    .put("seed", 0)
                    .put("offset", 0.0);

            assertThat(pyMap.toPyKwargs())
                    .contains("seed=0")
                    .contains("offset=0.0");
        }

        @Test
        void shouldChainMultiplePutCalls() {
            var pyMap = Gym.builderMap()
                    .put("a", 1)
                    .put("b", 2)
                    .put("c", 3)
                    .put("d", 4)
                    .put("e", 5);

            assertEquals(5, pyMap.size());
            assertFalse(pyMap.isEmpty());
        }

        @Test
        void shouldGenerateComplexCarRacingParams() {
            var pyMap = Gym.builderMap()
                    .put("domain_randomize", true)
                    .put("continuous", true)
                    .put("lap_complete_percent", 0.95);

            var expected = "domain_randomize=True, continuous=True, lap_complete_percent=0.95";
            assertEquals(expected, pyMap.toPyKwargs());
        }

        @Test
        void shouldGenerateAtariParams() {
            var pyMap = Gym.builderMap()
                    .put("obs_type", "rgb")
                    .put("frameskip", 1)
                    .put("repeat_action_probability", 0.0)
                    .put("full_action_space", false);

            assertThat(pyMap.toPyKwargs())
                    .contains("obs_type=rgb")
                    .contains("frameskip=1")
                    .contains("repeat_action_probability=0.0")
                    .contains("full_action_space=False");
        }

        @Test
        void shouldGenerateMujocoParams() {
            var pyMap = Gym.builderMap()
                    .put("xml_file", "humanoid.xml")
                    .put("forward_reward_weight", 1.25)
                    .put("ctrl_cost_weight", 0.1)
                    .put("reset_noise_scale", 0.01);

            assertEquals(4, pyMap.size());
            assertThat(pyMap.toPyKwargs())
                    .contains("xml_file=humanoid.xml")
                    .contains("forward_reward_weight=1.25")
                    .contains("ctrl_cost_weight=0.1")
                    .contains("reset_noise_scale=0.01");
        }

        @Test
        void shouldWorkWithGymBuilder() {
            var script = Gym.builder()
                    .envName("CartPole-v1")
                    .params(Gym.builderMap()
                            .put("max_episode_steps", 500)
                            .put("continuous", false))
                    .generatePyEnvScript();

            assertThat(script).contains("gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500, continuous=False)");
        }

        @Test
        void shouldWorkWithEmptyParamsInGymBuilder() {
            var script = Gym.builder()
                    .envName("CartPole-v1")
                    .params(Gym.builderMap())
                    .generatePyEnvScript();

            assertThat(script).contains("gym.make('CartPole-v1', render_mode='rgb_array')");
            assertThat(script).doesNotContain("gym.make('CartPole-v1', render_mode='rgb_array',");
        }

        @Test
        void shouldNotIncludeParamsWhenNotSet() {
            var script = Gym.builder()
                    .envName("CartPole-v1")
                    .generatePyEnvScript();

            assertThat(script).contains("gym.make('CartPole-v1', render_mode='rgb_array')");
            assertThat(script).doesNotContain(", , ");
        }

        @Test
        void shouldHandleSpecialCharactersInStringValues() {
            var pyMap = Gym.builderMap()
                    .putStr("path", "/tmp/data")
                    .putStr("name", "test-env");

            assertThat(pyMap.toPyKwargs())
                    .contains("path='/tmp/data'")
                    .contains("name='test-env'");
        }

        @Test
        void shouldHandleScientificNotation() {
            var pyMap = Gym.builderMap()
                    .put("epsilon", 1e-8)
                    .put("learning_rate", 3e-4);

            assertThat(pyMap.toPyKwargs())
                    .containsPattern("epsilon=1\\.0E-8|epsilon=0\\.00000001")
                    .containsPattern("learning_rate=3\\.0E-4|learning_rate=0\\.0003");
        }

        @Test
        void shouldBuildCompleteCarRacingExample() {
            var pyMap = Gym.builderMap()
                    .put("domain_randomize", true)
                    .put("continuous", true)
                    .put("lap_complete_percent", 0.95)
                    .put("max_episode_steps", 1000);

            assertEquals(4, pyMap.size());

            var kwargs = pyMap.toPyKwargs();
            assertThat(kwargs).contains("domain_randomize=True");
            assertThat(kwargs).contains("continuous=True");
            assertThat(kwargs).contains("lap_complete_percent=0.95");
            assertThat(kwargs).contains("max_episode_steps=1000");
        }

        @Test
        void shouldConvertToPyDictFormat() {
            var pyMap = Gym.builderMap()
                    .put("seed", 42)
                    .put("render_fps", 60);

            assertEquals("{'seed': 42, 'render_fps': 60}", pyMap.toPyDict());
        }

        @Test
        void shouldHandleVeryLargeNumbers() {
            var pyMap = Gym.builderMap()
                    .put("max_steps", Integer.MAX_VALUE)
                    .put("large_value", 1_000_000_000);

            assertThat(pyMap.toPyKwargs())
                    .contains("max_steps=" + Integer.MAX_VALUE)
                    .contains("large_value=1000000000");
        }
    }
}
