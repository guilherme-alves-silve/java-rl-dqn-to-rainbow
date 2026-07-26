package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static br.com.guialves.rflr.algorithms.networks.DuelingQNetworkMLP.withMeanType;
import static br.com.guialves.rflr.algorithms.networks.DuelingQNetworkMLP.withType;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static org.junit.jupiter.api.Assertions.*;

class DuelingQNetworkMLPTest {

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
    void shouldHaveForwardShape() {
        int observation = 8;
        int actions = 4;
        var expectedShape = new Shape(1, actions);
        @Cleanup var net = withMeanType(observation, actions, manager);
        var input = manager.ones(new Shape(1, observation));
        var safeCopy = input.duplicate();
        var output = net.forward(input);
        assertEquals(expectedShape, output.getShape());
        assertEquals(input, safeCopy);
    }

    @Test
    void shouldHaveForwardBatchShape() {
        int batch = 32;
        int observation = 8;
        int actions = 4;
        var singleShape = new Shape(1, actions);
        var expectedBatchShape = new Shape(batch, actions);
        @Cleanup var net = withMeanType(observation, actions, manager);
        var input = manager.ones(new Shape(batch, observation));
        var safeCopy = input.duplicate();
        var output = net.forward(input);
        assertNotEquals(singleShape, output.getShape());
        assertEquals(expectedBatchShape, output.getShape());
        assertEquals(input, safeCopy);
    }

    @ParameterizedTest(name = "[{index}] size={arguments}")
    @EnumSource(DuelingQNetworkMLP.DuelingType.class)
    void shouldAvoidMemoryLeak(DuelingQNetworkMLP.DuelingType duelingType) {
        @Cleanup var sub = manager.newSubManager();
        int observation = 8;
        int actions = 4;
        int expectedEnd = 9;
        int expectedAfter = 10;

        var net = withType(observation, actions, sub, duelingType);
        var input = sub.randomNormal(0f, 1f, new Shape(1, observation), DataType.FLOAT32);
        int between = managedArrayCount(sub);
        assertEquals(expectedEnd, between);
        for (int i = 0; i < 10; ++i) {
            @Cleanup var output = net.forward(input);
            assertFalse(output.isReleased());
            int after = managedArrayCount(sub);
            assertEquals(expectedAfter, after);
        }
        assertEquals(expectedEnd, managedArrayCount(sub));
    }
}
