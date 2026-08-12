package br.com.guialves.rflr.fixture;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import lombok.Cleanup;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * One-time fixture that materialises a real DJL {@code .params} file
 * under {@code src/test/resources/models/} so the {@code AgentDQN}
 * load-from-resource tests have a stable artefact to load.
 *
 * <p>Run it manually once with:
 * <pre>
 *   mvn -q test-compile exec:java \
 *       -Dexec.classpathScope=test \
 *       -Dexec.mainClass=br.com.guialves.rflr.algorithms.dqn.AgentDQNResourceFixture
 * </pre>
 * (or just invoke main() from an IDE) and commit the
 * resulting {@code src/test/resources/models/dqn_trained_lunar_lander-0000.params}
 * file alongside the test sources.
 *
 * <p>The "trained" model is in fact a freshly-initialised network —
 * what matters for the load tests is that the bytes-on-disk survive a
 * save/load round-trip and that the agent can be constructed from
 * them, not that the weights encode a useful policy.
 */
public class AgentDQNResourceFixture {

    public static final String MODEL_NAME = "dqn_trained_lunar_lander";
    public static final Path RESOURCE_DIR = Paths.get("src", "test", "resources", "models");
    public static final Path PARAMS_FILE = RESOURCE_DIR.resolve(MODEL_NAME + "-0000.params");

    static void main() throws Exception {
        if (Files.exists(PARAMS_FILE)) {
            IO.println("Fixture already present at " + PARAMS_FILE.toAbsolutePath());
            return;
        }

        Files.createDirectories(RESOURCE_DIR);
        try (var parent = NDManager.newBaseManager()) {
            int observations = 8;  // LunarLander observation dim
            int actions = 4;       // LunarLander action count
            @Cleanup var net = new DeepQNetworkMLP(observations, actions, parent);
            net.save(RESOURCE_DIR, MODEL_NAME);
        }
        IO.println("Wrote fixture to " + PARAMS_FILE.toAbsolutePath());
    }
}
