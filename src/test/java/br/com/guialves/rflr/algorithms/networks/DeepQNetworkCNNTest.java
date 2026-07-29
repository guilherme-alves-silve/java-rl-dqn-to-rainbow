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

import java.io.IOException;
import java.nio.file.Path;

import static br.com.guialves.rflr.algorithms.networks.DuelingQNetworkMLP.withType;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static org.junit.jupiter.api.Assertions.*;

class DeepQNetworkCNNTest {

    public static final float SAFE_FLOAT_COMPARISON = 1e-6f;
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

        int channels = 4;
        int observations = 84;
        int actions = 6;

        try (var cnn = new DeepQNetworkCNN(channels, observations, actions, manager)) {
            var input = manager.randomUniform(0f, 1f, new Shape(1, channels, observations, observations));
            var output = cnn.forward(input);
            assertEquals(new Shape(1, actions), output.getShape());
        }
    }

    @Test
    void testForwardBatchInput() {

        int channels = 4;
        int observations = 84;
        int actions = 3;
        int batchobservations = 16;

        try (var cnn = new DeepQNetworkCNN(channels, observations, actions, manager)) {
            var batch = manager.randomUniform(0f, 1f, new Shape(batchobservations, channels, observations, observations));
            var output = cnn.forward(batch);
            assertEquals(new Shape(batchobservations, actions), output.getShape());
        }
    }

    @Test
    void testDeterministicForward() {

        int channels = 4;
        int observations = 84;
        int actions = 4;

        try (var cnn = new DeepQNetworkCNN(channels, observations, actions, manager)) {
            var input = manager.randomUniform(0f, 1f, new Shape(1, channels, observations, observations));
            var out1 = cnn.forward(input);
            var out2 = cnn.forward(input);
            assertEquals(out1, out2);
        }
    }

    @Test
    void testSaveAndLoad(@TempDir Path tempDir) throws IOException {

        int channels = 4;
        int observations = 84;
        int actions = 5;
        String prefix = "cnn_test_model";

        var input = manager.randomUniform(0f, 1f, new Shape(1, channels, observations, observations));

        @Cleanup var cnn = new DeepQNetworkCNN(channels, observations, actions, manager);
        var originalOutput = cnn.forward(input).duplicate();
        cnn.save(tempDir, prefix);

        @Cleanup var loaded = new DeepQNetworkCNN(channels, observations, actions,
                tempDir, prefix, manager);
        var loadedOutput = loaded.forward(input);
        assertEquals(originalOutput.getShape(), loadedOutput.getShape());
        assertTrue(originalOutput.allClose(
                loadedOutput,
                SAFE_FLOAT_COMPARISON,
                SAFE_FLOAT_COMPARISON,
                false
        ));
    }

    @Test
    void testOutputNotNull() {

        int channels = 3;
        int observations = 64;
        int actions = 2;

        try (var cnn = new DeepQNetworkCNN(channels, observations, actions, manager)) {
            var input = manager.zeros(new Shape(1, channels, observations, observations));
            var output = cnn.forward(input);
            assertNotNull(output);
            assertEquals(actions, output.getShape().get(1));
        }
    }

    @Test
    void shouldAvoidMemoryLeak() {
        int channels = 3;
        int observations = 64;
        int actions = 2;
        int expectedEnd = 11;
        int expectedAfter = 12;

        @Cleanup var net = new DeepQNetworkCNN(channels, observations, actions, manager);
        var input = manager.randomNormal(0f, 1f, new Shape(1, channels, observations, observations), DataType.FLOAT32);
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
