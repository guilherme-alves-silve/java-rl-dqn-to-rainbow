package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.AdamW;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.fixture.AgentDQNResourceFixture;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the "load a pre-trained {@code DeepQNetworkMLP} from
 * {@code src/test/resources/}" path of {@link AgentDQN}.
 *
 * <p>The {@code .params} file is generated once by
 * {@link AgentDQNResourceFixture} and committed to the repo so these
 * tests have a stable artifact to load from. The file is a real
 * DJL save: same bytes-on-disk the agent would persist via
 * {@code AbstractAgent#save(Path, String)} after a training run.
 */
@DisplayName("AgentDQN: load pre-trained DeepQNetworkMLP from resources")
class AgentDQNLoadModelTest {

    private static final int OBSERVATIONS = 8;   // LunarLander
    private static final int ACTIONS = 4;        // LunarLander

    private NDManager parent;

    @BeforeAll
    static void verifyFixturePresent() {
        // Sanity-check the committed artifact before the tests run.
        assertTrue(Files.exists(AgentDQNResourceFixture.PARAMS_FILE),
                "Missing fixture at " + AgentDQNResourceFixture.PARAMS_FILE
                        + " - run AgentDQNResourceFixture#main once and commit the file.");
    }

    @BeforeEach
    void setUp() {
        parent = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        parent.close();
    }

    @Test
    @DisplayName("DeepQNetworkMLP loads cleanly from the resource file")
    void deepQNetworkMLP_loadsFromResources() {
        var modelPath = AgentDQNResourceFixture.RESOURCE_DIR;

        @Cleanup var net = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS,
                modelPath, AgentDQNResourceFixture.MODEL_NAME, parent);

        @Cleanup var input = parent.ones(new Shape(1, OBSERVATIONS));
        var output = net.forward(new NDList(input));
        assertEquals(new Shape(1, ACTIONS), output.singletonOrThrow().getShape());
        assertNotNull(output.singletonOrThrow());
    }

    @Test
    @DisplayName("Save -> load round-trip produces identical Q-values")
    void shouldLoadModelAndMatchesOriginalSaveBytes() throws Exception {
        @Cleanup var original = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, parent);
        Path tmp = Files.createTempDirectory("dqn-load-test");
        try {
            String prefix = "dqn_load_test";
            original.save(tmp, prefix);

            @Cleanup var loaded = new DeepQNetworkMLP(OBSERVATIONS, ACTIONS,
                    tmp, prefix, parent);

            @Cleanup var input = parent.ones(new Shape(1, OBSERVATIONS));
            NDArray fromOriginal = original.forward(input);
            NDArray fromLoaded  = loaded.forward(input);
            assertEquals(fromOriginal.getShape(), fromLoaded.getShape());
            assertEquals(fromOriginal, fromLoaded);
        } finally {
            // best-effort cleanup of the temp file we wrote
            try (var walker = Files.walk(tmp)) {
                    walker.sorted((a, b) -> b.getNameCount() - a.getNameCount())
                        .forEach(p -> {
                            try {
                                Files.deleteIfExists(p);
                            } catch (Exception ignored) {
                                // Ignored
                            }
                        });
            }
        }
    }

    @Test
    @DisplayName("AgentDQN can be built around a network loaded from resources")
    void agentDQN_isConstructibleWithLoadedNetwork() {
        var modelPath = AgentDQNResourceFixture.RESOURCE_DIR;

        Optimizer optimizer = AdamW.builder()
                .optLearningRateTracker(Tracker.fixed(0.0005f))
                .build();

        @Cleanup var agent = new AgentDQN(
                /* epsilon        */ 0.0f,   // greedy - force the net, not random
                /* updateQTargetN */ 1000,
                /* minEpsilon     */ 0.0f,
                /* epsilonDecay   */ 0.0f,
                /* gamma          */ 0.99f,
                new MockDiscreteEnv(OBSERVATIONS, ACTIONS),
                optimizer,
                parent,
                () -> new DeepQNetworkMLP(OBSERVATIONS, ACTIONS, modelPath,
                        AgentDQNResourceFixture.MODEL_NAME, parent),
                new PlotTrackers(),
                /* debugMemoryLeak */ false
        );

        @Cleanup var parentPerCall = br.com.guialves.rflr.djlutils.DJLMemoryManagement
                .subMgr(parent, "load-test-call");
        @Cleanup var state = parentPerCall.ones(new Shape(OBSERVATIONS));
        var action = agent.selectAction(state);
        assertNotNull(action);
        long chosen = action.value();
        assertTrue(chosen >= 0 && chosen < ACTIONS,
                "Action " + chosen + " is outside the configured [0, " + ACTIONS + ") range");
    }

    /**
     * Minimal {@link IEnv} that just keeps cycling for the duration of
     * a single {@code selectAction} call. The agent only ever invokes
     * {@code actionSpaceType()} on the env during selection, so this
     * is enough to satisfy the constructor.
     */
    private static final class MockDiscreteEnv implements IEnv {
        private final int obsDim;
        private final int nActions;
        private final ActionSpaceType type = ActionSpaceType.DISCRETE;

        MockDiscreteEnv(int obsDim, int nActions) {
            this.obsDim = obsDim;
            this.nActions = nActions;
        }

        @Override
        public boolean closed() {
            return false;
        }

        @Override
        public boolean scalarObservation() {
            return true;
        }

        @Override
        public ActionSpaceType actionSpaceType() {
            return type;
        }

        @Override
        public String actionSpaceStr() {
            return "Discrete(" + nActions + ")";
        }

        @Override
        public String observationSpaceStr() {
            return "Box(" + obsDim + ")";
        }

        @Override
        public ActionSpaceType.ActionResult actionSpaceSample() {
            return type.get(0);
        }

        @Override
        public EnvResetResult reset(NDManager manager) {
            var state = manager.zeros(new Shape(obsDim));
            return new EnvResetResult(state,
                    Collections.emptyMap());
        }

        @Override
        public EnvStepResult step(ActionSpaceType.ActionResult action, NDManager sub) {
            @Cleanup var next = sub.zeros(new Shape(obsDim));
            return new EnvStepResult(0.0, false, false, Collections.emptyMap(), next);
        }

        @Override
        public BufferedImage render() {
            return null;
        }

        @Override
        public void close() {
            // Ignored
        }
    }
}
