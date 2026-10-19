package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.gymnasium4j.wrappers.DelayObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.FrameStackObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.GrayscaleObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.MaxAndSkipObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.NormalizeObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Coverage for {@link VectorGym} and its {@link VectorGym.VectorEnvBuilder}.
 *
 * <p>Every test here asserts on the <em>generated Python source</em>, so the suite never needs a
 * Python interpreter. The generated vector script is a pure function of the builder state.
 */
class VectorGymTest {

    private static final String VAR_ENV_PATTERN = "env_[0-9a-f]{32}";
    private static final String VIDEO = "/tmp/video";

    @Test
    void shouldExposeVectorEnvBuilder() {
        assertThat(VectorGym.builder()).isInstanceOf(VectorGym.VectorEnvBuilder.class);
    }

    @Test
    void shouldDefaultToSingleSyncEnvPerName() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(")
                .contains("for idx in range(1)]");
    }

    @Test
    void shouldDefaultFunctionToCallToMakeEnv() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("def make_env(idx: int, env_wrappers: tuple[str, dict]):")
                .contains("make_env(idx, env_wrappers) for env_wrappers in")
                .doesNotContain("make_retro_env");
    }

    @Test
    void shouldGenerateUniqueVarEnvCodePerBuilder() {
        var first = VectorGym.builder().envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1)).generatePyEnvScript();
        var second = VectorGym.builder().envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1)).generatePyEnvScript();

        assertThat(first).isNotEqualTo(second);
    }

    @Test
    void shouldReuseVarEnvCodeWithinSameBuilder() {
        var builder = VectorGym.builder().envsNames(List.of("CartPole-v1"));

        var scriptA = builder.add(new DelayObservation(1)).generatePyEnvScript();
        var scriptB = builder.generatePyEnvScript();

        var varEnv = extractVarEnvCode(scriptA);
        assertThat(varEnv).isEqualTo(extractVarEnvCode(scriptB));
        assertThat(scriptB).contains(varEnv + " = ");
    }

    private static String extractVarEnvCode(String script) {
        return java.util.regex.Pattern
                .compile("env_([0-9a-f]{32}) = gym\\.vector\\.")
                .matcher(script)
                .results()
                .findFirst()
                .orElseThrow(() -> new AssertionError("no vector env assignment found in script"))
                .group(1);
    }

    @Nested
    @DisplayName("numberOfEnvs validation")
    class NumberOfEnvsValidation {

        @Test
        void shouldExpandEveryEnvNameByNumberOfEnvs() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1", "MountainCar-v0"))
                    .numberOfEnvs(4)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            assertThat(script)
                    .contains("for env_wrappers in [('CartPole-v1', env_params), ('MountainCar-v0', env_params)]")
                    .contains("for idx in range(4)]");
        }

        @Test
        void shouldAcceptSingleEnv() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .numberOfEnvs(1)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            assertThat(script).contains("for idx in range(1)]");
        }

        @Test
        void shouldRejectZeroEnvs() {
            assertThatThrownBy(() -> VectorGym.builder().numberOfEnvs(0))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessage("numberOfEnvs must be greater than 0");
        }

        @Test
        void shouldRejectNegativeEnvs() {
            assertThatThrownBy(() -> VectorGym.builder().numberOfEnvs(-1))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessage("numberOfEnvs must be greater than 0");
        }

        @Test
        void shouldRejectZeroEnvsEvenWithLargeNegativeValue() {
            assertThatThrownBy(() -> VectorGym.builder().numberOfEnvs(Integer.MIN_VALUE))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessage("numberOfEnvs must be greater than 0");
        }
    }

    @Test
    void shouldExpandEnvNameIntoSingleEntryEnvList() {
        var script = VectorGym.builder()
                .envName("CartPole-v1")
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("for env_wrappers in [('CartPole-v1', env_params)]")
                .doesNotContain("envName")
                .doesNotContain("null");
    }

    @Test
    void shouldKeepEnvNameOrderInEnvList() {
        var script = VectorGym.builder()
                .envsNames(List.of("A-v0", "B-v0", "C-v0"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains(
                "for env_wrappers in [('A-v0', env_params), ('B-v0', env_params), ('C-v0', env_params)]");
    }

    @Test
    void shouldGenerateSyncVectorEnvWhenAsyncDisabled() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .async(false)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(")
                .doesNotContain("AsyncVectorEnv");
    }

    @Test
    void shouldGenerateAsyncVectorEnvWhenAsyncEnabled() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .async(true)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.AsyncVectorEnv\\(")
                .doesNotContain("SyncVectorEnv");
    }

    @Test
    void shouldKeepAsyncVectorEnvBodyIdenticalToSync() {
        // The vector setters return VectorEnvBuilder, so the fluent chain is preserved end to end.
        var sync = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .numberOfEnvs(3)
                .async(false)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        var async = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .numberOfEnvs(3)
                .async(true)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        var sharedBody = "for env_wrappers in [('CartPole-v1', env_params)] for idx in range(3)";
        assertThat(sync).contains(sharedBody).contains("SyncVectorEnv");
        assertThat(async).contains(sharedBody).contains("AsyncVectorEnv");
    }

    @Test
    void shouldSupportFullFluentChainInAnyOrder() {
        // Guards the builder signatures: every vector setter must return VectorEnvBuilder,
        // otherwise the chain degrades to EnvBuilder and loses async/numberOfEnvs/envsNames.
        var script = VectorGym.builder()
                .async(true)
                .numberOfEnvs(2)
                .envsNames(List.of("CartPole-v1"))
                .envName("MountainCar-v0")
                .importLib("shimmy")
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("import gymnasium as gym, shimmy\n")
                .containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.AsyncVectorEnv\\(")
                .contains("for env_wrappers in [('MountainCar-v0', env_params)] for idx in range(2)]");
    }

    @Test
    void shouldOmitVectorKwargsWhenParamsEmpty() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .params(Gym.builderMap())
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(.*\\)$")
                .doesNotContain("context=")
                .doesNotContain("autoreset=");
    }

    @Test
    void shouldAppendVectorKwargsWhenParamsProvided() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .params(Gym.builderMap()
                        .putStr("context", "spawn")
                        .put("autoreset", false))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(
                VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(.*, context='spawn', autoreset=False\\)$");
    }

    @Test
    void shouldAppendVectorKwargsToAsyncVectorEnv() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .async(true)
                .params(Gym.builderMap().putStr("context", "forkserver"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(
                VAR_ENV_PATTERN + " = gym\\.vector\\.AsyncVectorEnv\\(.*, context='forkserver'\\)$");
    }

    @Nested
    @DisplayName("Wrapper kwargs rendering")
    class WrapperKwargs {

        @Test
        void shouldRenderSingleWrapperEnvParams() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            assertThat(script).contains("env_params = [ (DelayObservation , {'delay': 1}) ]");
        }

        @Test
        void shouldRenderMultipleWrappersInDeclarationOrder() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new DelayObservation(1),
                            new GrayscaleObservation(true),
                            new FrameStackObservation(4))
                    .generatePyEnvScript();

            assertThat(script).contains(
                    "env_params = [ (DelayObservation , {'delay': 1}), " +
                            "(GrayscaleObservation , {'keep_dim': True}), " +
                            "(FrameStackObservation , {'stack_size': 4}) ]");
        }

        @Test
        void shouldRenderVarargsOnlyWrapper() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new RecordEpisodeStatistics())
                    .generatePyEnvScript();

            assertThat(script).contains("env_params = [ (RecordEpisodeStatistics , {  }) ]");
        }

        @Test
        void shouldRenderRecordVideoKwargs() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new RecordVideo(VIDEO, -1, 1000, 500, "test-prefix", 30, true, 50))
                    .generatePyEnvScript();

            assertThat(script).contains("'video_folder': '/tmp/video'")
                    .contains("'episode_trigger': None")
                    .contains("'step_trigger': lambda t: t % 1000 == 0")
                    .contains("'video_length': 500")
                    .contains("'name_prefix': 'test-prefix'")
                    .contains("'fps': 30")
                    .contains("'disable_logger': True")
                    .contains("'gc_trigger': 50");
        }

        @Test
        void shouldRenderEpisodeTriggerLambda() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new RecordVideo(VIDEO, 10, 1000, 500, "prefix", 30, false, -1))
                    .generatePyEnvScript();

            assertThat(script).contains("'episode_trigger': lambda t: t % 10 == 0")
                    .doesNotContain("'disable_logger'")
                    .doesNotContain("'gc_trigger'");
        }

        @Test
        void shouldImportEveryWrapperFromGymnasium() {
            var script = VectorGym.builder()
                    .envsNames(List.of("CartPole-v1"))
                    .add(new DelayObservation(1),
                            new GrayscaleObservation(false),
                            new FrameStackObservation(4),
                            new RecordEpisodeStatistics(50, "stats"))
                    .generatePyEnvScript();

            assertThat(script).contains(
                    "from gymnasium.wrappers import DelayObservation, GrayscaleObservation, " +
                            "FrameStackObservation, RecordEpisodeStatistics");
        }
    }

    @Test
    void shouldGenerateBaseGymImport() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).startsWith("import gymnasium as gym\n");
    }

    @Test
    void shouldAggregateAdditionalImports() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .importLib("ale_py")
                .importLib("shimmy")
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).startsWith("import gymnasium as gym, ale_py, shimmy\n");
    }

    @Test
    void shouldKeepBaseMakeEnvDefinitionBody() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("env = gym.make(id=env_name, render_mode='rgb_array')")
                .contains("env = wrapper(env, **params)")
                .contains("return _init");
    }

    @Test
    void shouldOverrideFunctionToCall() {
        var builder = VectorGym.builder();
        builder.envsNames(List.of("CartPole-v1")).add(new DelayObservation(1));
        builder.setFunctionToCall("make_custom_env");

        var script = builder.generatePyEnvScript();

        assertThat(script).contains("make_custom_env(idx, env_wrappers) for env_wrappers in [('CartPole-v1', env_params)] for idx in range(1)")
                .doesNotContain("make_env(idx, env_wrappers) for env_wrappers in");
    }

    @Test
    void shouldKeepVideoFolderIsolationSnippet() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new RecordVideo(VIDEO))
                .generatePyEnvScript();

        assertThat(script).contains("if 'video_folder' in wrapper_params:")
                .contains("params['video_folder'] = f\"{params['video_folder']}/{env_name}{idx}\"");
    }

    @Test
    void shouldShareTheSameEnvParamsAcrossEveryEnvAndIndex() {
        var script = VectorGym.builder()
                .envsNames(List.of("A-v0", "B-v0"))
                .numberOfEnvs(3)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("for env_wrappers in " +
                "[('A-v0', env_params), ('B-v0', env_params)] for idx in range(3)");
    }

    @Test
    void shouldSupportVectorEnvWithoutWrappers() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .generatePyEnvScript();

        assertThat(script).contains("env_params = [ ]")
                .doesNotContain("from gymnasium.wrappers import")
                .containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(.*\\)$");
    }

    @Test
    void shouldBuildVectorEnvFromEnvName() {

        try (var env = VectorGym.builder()
                .envName("CartPole-v1")
                .add(new DelayObservation(1))
                .build()) {
            assertThat(env.envName()).isEqualTo("CartPole-v1");
        }
    }

    @Test
    void shouldRenderMaxAndSkipSkipKey() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new MaxAndSkipObservation(4))
                .generatePyEnvScript();

        assertThat(script).contains("env_params = [ (MaxAndSkipObservation , {'skip': 4}) ]")
                .doesNotContain("keep_dim");
    }

    @Test
    void shouldRenderNormalizeEpsilonKey() {
        var script = VectorGym.builder()
                .envsNames(List.of("CartPole-v1"))
                .add(new NormalizeObservation())
                .generatePyEnvScript();

        assertThat(script).contains("env_params = [ (NormalizeObservation , {'epsilon': 1.0E-8}) ]")
                .doesNotContain("keep_dim");
    }
}
