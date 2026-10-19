package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.gymnasium4j.wrappers.DelayObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.FrameStackObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.GrayscaleObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * Coverage for {@link StableRetroGym} and its {@link StableRetroGym.StableRetroEnvBuilder}.
 *
 * <p>Every test here asserts on the <em>generated Python source</em>, so the suite never needs a
 * Python interpreter or an installed Stable Retro ROM.
 */
class StableRetroGymTest {

    private static final String VAR_ENV_PATTERN = "env_[0-9a-f]{32}";
    private static final List<String> GAMES = List.of("MegaMan-Nes-v0", "MegaMan2-Nes-v0");
    private static final String VIDEO = "/tmp/video";

    // ------------------------------------------------------------------
    // Builder construction
    // ------------------------------------------------------------------

    @Test
    void shouldExposeStableRetroEnvBuilder() {
        assertThat(StableRetroGym.builder()).isInstanceOf(StableRetroGym.StableRetroEnvBuilder.class);
    }

    @Test
    void shouldBeAVectorGym() {
        assertThat(StableRetroGym.builder()).isInstanceOf(VectorGym.VectorEnvBuilder.class);
    }

    // ------------------------------------------------------------------
    // Retro-specific script generation
    // ------------------------------------------------------------------

    @Test
    void shouldImportStableRetroAlongsideGymnasium() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).startsWith("import gymnasium as gym, stable_retro as retro\n");
    }

    @Test
    void shouldAggregateExtraImportsAfterRetro() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .importLib("shimmy")
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).startsWith("import gymnasium as gym, stable_retro as retro, shimmy\n");
    }

    @Test
    void shouldDefineMakeRetroEnv() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("def make_retro_env(idx: int, env_wrappers: tuple[str, dict]):")
                .contains("env_name, wrappers = env_wrappers")
                .contains("return _init");
    }

    @Test
    void shouldCreateEnvThroughRetroMakeWithAllIntegrations() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains(
                        "env = retro.make(game=env_name, render_mode='rgb_array', inttype=retro.data.Integrations.ALL)")
                .doesNotContain("gym.make(id=env_name, render_mode='rgb_array')\n    for wrapper");
    }

    @Test
    void shouldInvokeMakeRetroEnvInVectorCall() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .numberOfEnvs(3)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(
                        VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(" +
                                "\\[make_retro_env\\(idx, env_wrappers\\) for env_wrappers in " +
                                "\\[\\('MegaMan\\-Nes\\-v0', env_params\\), \\('MegaMan2\\-Nes\\-v0', env_params\\)\\] " +
                                "for idx in range\\(3\\)\\]")
                .doesNotContain("=gym.vector.SyncVectorEnv([make_env(idx");
    }

    @Test
    void shouldKeepWrapperApplicationLoopInRetroEnv() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("for wrapper, wrapper_params in wrappers:")
                .contains("params = dict(wrapper_params)")
                .contains("env = wrapper(env, **params)");
    }

    @Test
    void shouldKeepPerEnvVideoFolderIsolation() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new RecordVideo(VIDEO))
                .generatePyEnvScript();

        assertThat(script).contains("if 'video_folder' in wrapper_params:")
                .contains("params['video_folder'] = f\"{params['video_folder']}/{env_name}{idx}\"");
    }

    // ------------------------------------------------------------------
    // Wrapper kwargs
    // ------------------------------------------------------------------

    @Test
    void shouldRenderSingleWrapperEnvParams() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains("env_params = [ (DelayObservation , {'delay': 1}) ]");
    }

    @Test
    void shouldRenderMultipleWrappersInDeclarationOrder() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1),
                        new GrayscaleObservation(true),
                        new FrameStackObservation(4),
                        new RecordEpisodeStatistics(50, "retro_stats"))
                .generatePyEnvScript();

        assertThat(script).contains(
                        "env_params = [ (DelayObservation , {'delay': 1}), " +
                                "(GrayscaleObservation , {'keep_dim': True}), " +
                                "(FrameStackObservation , {'stack_size': 4}), " +
                                "(RecordEpisodeStatistics , { 'buffer_length': 50, 'stats_key': 'retro_stats' }) ]")
                .contains("from gymnasium.wrappers import DelayObservation, GrayscaleObservation, " +
                        "FrameStackObservation, RecordEpisodeStatistics");
    }

    @Test
    void shouldRenderRecordVideoKwargs() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
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
    void shouldShareSameEnvParamsAcrossEveryGameAndIndex() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .numberOfEnvs(7)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).contains(
                        "make_retro_env(idx, env_wrappers) for env_wrappers in " +
                                "[('MegaMan-Nes-v0', env_params), ('MegaMan2-Nes-v0', env_params)] " +
                                "for idx in range(7)]")
                .containsOnlyOnce("env_params = [");
    }

    // ------------------------------------------------------------------
    // sync / async / params inherited from VectorGym
    // ------------------------------------------------------------------

    @Test
    void shouldDefaultToSyncVectorEnv() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(")
                .doesNotContain("AsyncVectorEnv");
    }

    @Test
    void shouldGenerateAsyncVectorEnv() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .async(true)
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.AsyncVectorEnv\\(");
    }

    @Test
    void shouldAppendVectorKwargsWhenParamsProvided() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .params(Gym.builderMap().putStr("context", "spawn"))
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(.*, context='spawn'\\)$");
    }

    @Test
    void shouldOmitVectorKwargsWhenParamsEmpty() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .params(Gym.builderMap())
                .add(new DelayObservation(1))
                .generatePyEnvScript();

        assertThat(script).doesNotContain("context=");
    }

    @Nested
    @DisplayName("Script structure")
    class ScriptStructure {

        @Test
        void shouldEmitImportsBeforeDefinitionsAndAssignment() {
            var script = StableRetroGym.builder()
                    .envsNames(GAMES)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            var importIdx = script.indexOf("import gymnasium as gym, stable_retro as retro");
            var wrapperImportIdx = script.indexOf("from gymnasium.wrappers import");
            var defIdx = script.indexOf("def make_retro_env");
            var paramsIdx = script.indexOf("env_params = [");
            var assignIdx = script.lastIndexOf("= gym.vector.");

            assertThat(importIdx).isZero();
            assertThat(wrapperImportIdx).isGreaterThan(importIdx);
            assertThat(defIdx).isGreaterThan(wrapperImportIdx);
            assertThat(paramsIdx).isGreaterThan(defIdx);
            assertThat(assignIdx).isGreaterThan(paramsIdx);
        }

        @Test
        void shouldEndWithVectorEnvAssignment() {
            var script = StableRetroGym.builder()
                    .envsNames(GAMES)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            assertThat(script.strip()).endsWith("for idx in range(1)])");
        }

        @Test
        void shouldUseUniqueVarEnvCodePerBuilder() {
            var first = StableRetroGym.builder().envsNames(GAMES)
                    .add(new DelayObservation(1)).generatePyEnvScript();
            var second = StableRetroGym.builder().envsNames(GAMES)
                    .add(new DelayObservation(1)).generatePyEnvScript();

            assertThat(first).isNotEqualTo(second);
        }

        @Test
        void shouldNotEmitTheDeadBaseMakeEnvDefinition() {
            var script = StableRetroGym.builder()
                    .envsNames(GAMES)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            assertThat(script).doesNotContain("def make_env(idx: int, env_wrappers: tuple[str, dict]):")
                    .doesNotContain("env = gym.make(id=env_name, render_mode='rgb_array')")
                    .contains("def make_retro_env(idx: int, env_wrappers: tuple[str, dict]):");
        }

        @Test
        void shouldGenerateParsableIndentStructure() {
            var script = StableRetroGym.builder()
                    .envsNames(GAMES)
                    .add(new DelayObservation(1))
                    .generatePyEnvScript();

            var occurrences = script.lines()
                    .filter(line -> line.startsWith("    def _init():"))
                    .count();

            assertThat(occurrences).isEqualTo(1);
            assertThat(script.lines().filter(line -> line.equals("    return _init")).count()).isEqualTo(1);
        }
    }

    @Test
    void shouldSupportRetroVectorEnvWithoutWrappers() {
        var script = StableRetroGym.builder()
                .envsNames(GAMES)
                .generatePyEnvScript();

        assertThat(script).contains("env_params = [ ]")
                .doesNotContain("from gymnasium.wrappers import")
                .containsPattern(VAR_ENV_PATTERN + " = gym\\.vector\\.SyncVectorEnv\\(.*\\)$");
    }

    /**
     * Building a Stable Retro env imports the {@code stable_retro} Python package, which only
     * installs on Linux. The assumption makes the test run automatically under WSL2 and skip on
     * Windows, instead of relying on an annotation that goes stale as soon as you switch host.
     */
    @Test
    void shouldBuildRetroVectorEnvFromEnvName() {
        var onWindows = System.getProperty("os.name").toLowerCase(Locale.ROOT).startsWith("windows");
        assumeFalse(onWindows,
                "requires the stable_retro Python package, which only installs on Linux/WSL2");

        try (var env = StableRetroGym.builder()
                .envName("MegaMan-Nes-v0")
                .add(new DelayObservation(1))
                .build()) {
            assertThat(env.envName()).isEqualTo("MegaMan-Nes-v0");
        }
    }
}