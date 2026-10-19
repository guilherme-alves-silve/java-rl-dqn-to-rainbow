package br.com.guialves.rflr.gymnasium4j.stableretro;

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
import br.com.guialves.rflr.gymnasium4j.Gym;

public class StableRetroTest {

    @Test
    void should() {
        var envs = List.of("MegaMan-Nes-v0", "MegaMan2-Nes-v0");
        var script = StableRetro.builder()
                .envsNames(envs)
                .params(StableRetro.builderMap()
                        .put("context", "spawn"))
                .add(new DelayObservation(1),
                        new RecordVideo("/tmp/video", 10,
                                1000, 500, "test-prefix",
                                30, true, 50))
                .generatePyEnvScript();

        assertThat(script).contains("import gymnasium as gym, stable_retro as retro")
                .contains("from gymnasium.wrappers import DelayObservation, RecordVideo")
                .contains("env_params = [ (DelayObservation, {'delay': 1}), (RecordVideo, {'video_folder': '/tmp/video', 'video_length': 500, 'fps': 30}) ]")
                .containsPattern("env_[0-9a-f]{32} = AsyncVectorEnv([make_env(idx, env_wrappers) for env_wrappers in " +
                        "[('MegaMan-Nes-v0', env_params), ('MegaMan2-Nes-v0', env_params)] for idx in range(n_envs)], context='spawn')");
    }
}
