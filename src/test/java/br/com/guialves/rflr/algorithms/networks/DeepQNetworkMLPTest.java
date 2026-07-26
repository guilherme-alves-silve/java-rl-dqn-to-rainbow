package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

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
        int observations = 4;
        int actions = 2;

        @Cleanup var net = new DeepQNetworkMLP(observations, actions, manager);
        var input = manager.ones(new Shape(1, observations));
        var output = net.forward(input);
        assertEquals(new Shape(1, actions), output.getShape());
    }

    @Test
    void testForwardBatchInput() {
        int observations = 4;
        int actions = 3;
        int batchSize = 8;

        @Cleanup var net = new DeepQNetworkMLP(observations, actions, manager);
        var batch = manager.randomUniform(0f, 1f, new Shape(batchSize, observations));
        var output = net.forward(batch);
        assertEquals(new Shape(batchSize, actions), output.getShape());
    }

    @Test
    void testModelSaveAndLoad(@TempDir Path tempDir) {

        int observations = 4;
        int actions = 2;
        String prefix = "dqn_test";

        var input = manager.randomUniform(0f, 1f, new Shape(1, observations));
        NDArray originalOutput;
        try (DeepQNetworkMLP dqn = new DeepQNetworkMLP(observations, actions, manager)) {
            originalOutput = dqn.forward(input).duplicate();
            dqn.save(tempDir, prefix);
        }

        @Cleanup var loaded = new DeepQNetworkMLP(observations, actions, tempDir, prefix, manager);
        var loadedOutput = loaded.forward(input);
        assertEquals(originalOutput.getShape(), loadedOutput.getShape());
        assertEquals(originalOutput.getDataType(), loadedOutput.getDataType());
        assertEquals(originalOutput, loadedOutput);
    }

    @Test
    void testMultipleForwardConsistency() {

        int observations = 4;
        int actions = 2;

        @Cleanup var net = new DeepQNetworkMLP(observations, actions, manager);
        var input = manager.randomUniform(0f, 1f, new Shape(1, observations));
        var out1 = net.forward(input);
        var out2 = net.forward(input);
        assertEquals(out1, out2);
    }

    @Test
    void shouldAvoidMemoryLeak() {
        int observations = 8;
        int actions = 4;
        int expectedEnd = 7;
        int expectedAfter = 8;

        @Cleanup var net = new DeepQNetworkMLP(observations, actions, manager);
        var input = manager.randomNormal(0f, 1f, new Shape(1, observations), DataType.FLOAT32);
        int between = managedArrayCount(manager);
        assertEquals(expectedEnd, between);
        for (int i = 0; i < 10; ++i) {
            @Cleanup var output = net.forward(input);
            assertFalse(output.isReleased());
            int after = managedArrayCount(manager);
            assertEquals(expectedAfter, after);
        }
        assertEquals(expectedEnd, managedArrayCount(manager));
    }
}
