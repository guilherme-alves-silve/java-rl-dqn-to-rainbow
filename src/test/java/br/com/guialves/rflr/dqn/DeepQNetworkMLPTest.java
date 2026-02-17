package br.com.guialves.rflr.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeepQNetworkMLPTest {

    private static NDManager manager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterAll
    static void cleanup() {
        manager.close();
    }

    @Test
    void testForwardSingleInput() {
        int obs = 4;
        int actions = 2;

        try (var dqn = new DeepQNetworkMLP(obs, actions, manager)) {
            var input = manager.ones(new Shape(1, obs));
            var output = dqn.forward(input);
            assertEquals(new Shape(1, actions), output.getShape());
        }
    }

    @Test
    void testForwardBatchInput() {
        int obs = 4;
        int actions = 3;
        int batchSize = 8;

        try (var dqn = new DeepQNetworkMLP(obs, actions, manager)) {
            var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, obs));
            var output = dqn.forward(batch);
            assertEquals(new Shape(batchSize, actions), output.getShape());
        }
    }

    @Test
    void testModelSaveAndLoad(@TempDir Path tempDir) {

        int obs = 4;
        int actions = 2;
        String prefix = "dqn_test";

        var input = manager.randomUniform(0f, 1f, new Shape(1, obs));
        NDArray originalOutput;

        try (DeepQNetworkMLP dqn = new DeepQNetworkMLP(obs, actions, manager)) {
            originalOutput = dqn.forward(input).duplicate();
            dqn.save(tempDir, prefix);
        }

        try (var loaded = new DeepQNetworkMLP(obs, actions, tempDir, prefix, manager)) {
            var loadedOutput = loaded.forward(input);
            assertEquals(originalOutput.getShape(), loadedOutput.getShape());
            assertEquals(originalOutput.getDataType(), loadedOutput.getDataType());
            assertEquals(originalOutput, loadedOutput);
        }
    }

    @Test
    void testMultipleForwardConsistency() {

        int obs = 4;
        int actions = 2;

        try (var dqn = new DeepQNetworkMLP(obs, actions, manager)) {
            var input = manager.randomUniform(0f, 1f, new Shape(1, obs));
            var out1 = dqn.forward(input);
            var out2 = dqn.forward(input);
            assertEquals(out1, out2);
        }
    }
}
