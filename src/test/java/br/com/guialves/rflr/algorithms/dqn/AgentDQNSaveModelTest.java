package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.AdamW;
import ai.djl.training.tracker.Tracker;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.execs.RLConfig;
import br.com.guialves.rflr.fixture.MockDiscreteEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the model persistence of {@link AgentDQN}.
 * <p>
 * Atari environments are registered as {@code ALE/Galaxian-v5}. When the
 * environment name was used as-is to build the model file name, DJL resolved
 * the parameters file to {@code output_models/<algorithm>/ALE/<name>-0000.params}
 * and failed with {@link java.nio.file.NoSuchFileException}, because the nested
 * {@code ALE} folder was never created.
 */
@DisplayName("AgentDQN: save the model where it can actually be written")
class AgentDQNSaveModelTest {

    private static final int OBSERVATIONS = 8;
    private static final int ACTIONS = 4;
    private static final String ALGORITHM_NAME = "galaxian_noisy_nets_dqn";
    private static final String ENV_NAME = "ALE/Galaxian-v5";

    private NDManager parent;

    @BeforeEach
    void setUp() {
        parent = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        parent.close();
    }

    @Test
    @DisplayName("The model of an ALE environment is saved flat in the algorithm folder")
    void shouldSaveModelFlatInAlgorithmFolder(@TempDir Path tempDir) throws Exception {
        var target = tempDir.resolve("output_models").resolve(ALGORITHM_NAME);
        var config = config(target);
        var modelFileName = config.modelFileName();

        try (var agent = newAgent()) {
            assertDoesNotThrow(() -> agent.save(target, modelFileName));
        }

        assertTrue(Files.exists(target.resolve(modelFileName + "-0000.params")),
                "The parameters file was not created in " + target);
        assertFalse(Files.exists(target.resolve("ALE")),
                "The environment name must not create a nested folder");
    }

    @Test
    @DisplayName("A model name carrying a path separator still finds its folder")
    void shouldCreateFolderOfNestedModelName(@TempDir Path tempDir) throws Exception {
        var target = tempDir.resolve("output_models").resolve(ALGORITHM_NAME);
        var modelFileName = ENV_NAME + "_1_" + ALGORITHM_NAME;

        try (var agent = newAgent()) {
            assertDoesNotThrow(() -> agent.save(target, modelFileName));
        }

        assertTrue(Files.isDirectory(target.resolve("ALE")),
                "The folder of the model name was not created");
        assertTrue(Files.exists(target.resolve(modelFileName + "-0000.params")),
                "The parameters file was not created in " + target);
    }

    private static RLConfig config(Path path) {
        return RLConfig.builder()
                .envName(ENV_NAME)
                .algorithmName(ALGORITHM_NAME)
                .path(path)
                .observations(OBSERVATIONS)
                .actions(ACTIONS)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .framesLimit(1_000_000)
                .build();
    }

    private AgentDQN newAgent() {
        var optimizer = AdamW.builder()
                .optLearningRateTracker(Tracker.fixed(0.0005f))
                .build();
        return new AgentDQN(
                /* epsilon        */ 0.0f,
                /* updateQTargetN */ 1000,
                /* minEpsilon     */ 0.0f,
                /* epsilonDecay   */ 0.0f,
                /* gamma          */ 0.99f,
                new MockDiscreteEnv(OBSERVATIONS, ACTIONS),
                optimizer,
                parent,
                () -> new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, parent),
                new PlotTrackers("test"),
                /* debugMemoryLeak */ false
        );
    }
}
