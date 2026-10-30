package br.com.guialves.rflr.execs;

import br.com.guialves.rflr.gymnasium4j.Gym;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static br.com.guialves.rflr.execs.RLConfig.Vectorized;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RLConfigTest {

    private static final String ALGORITHM_NAME = "enduro_noisy_nets_dqn";

    private static RLConfig.RLConfigBuilder base() {
        return RLConfig.builder()
                .envName("ALE/Enduro-v5")
                .runnerClass("AgentCNNDQNMain")
                .algorithmName(ALGORITHM_NAME)
                .observations(84)
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(5000)
                .batchSize(32)
                .framesLimit(1_000_000);
    }

    @Nested
    @DisplayName("Values provided to the builder must survive build()")
    class BuilderValues {

        @Test
        @DisplayName("channels provided by the builder is preserved")
        void shouldPreserveChannels() {
            var config = base().channels(4).build();
            assertEquals(4, config.channels());
        }

        @Test
        @DisplayName("channels defaults to 1 when the builder does not provide it")
        void shouldDefaultChannels() {
            var config = base().build();
            assertEquals(1, config.channels());
        }

        @Test
        @DisplayName("vectorized provided by the builder is preserved")
        void shouldPreserveVectorized() {
            var config = base().vectorized(Vectorized.ASYNC).build();
            assertEquals(Vectorized.ASYNC, config.vectorized());
            assertTrue(config.vectorized().isAsync());
        }

        @Test
        @DisplayName("envParams provided by the builder is preserved")
        void shouldPreserveEnvParams() {
            var config = base().envParams(Gym.builderMap().put("seed", 42)).build();
            assertEquals(1, config.envParams().size());
        }

        @Test
        @DisplayName("path provided by the builder is preserved")
        void shouldPreservePath() {
            var path = Path.of("./output_models/custom");
            var config = base().path(path).build();
            assertEquals(path, config.path());
        }

        @Test
        @DisplayName("path defaults to the algorithm output folder")
        void shouldDefaultPath() {
            var config = base().build();
            assertEquals(Path.of("./output_models/", ALGORITHM_NAME), config.path());
        }
    }
}
