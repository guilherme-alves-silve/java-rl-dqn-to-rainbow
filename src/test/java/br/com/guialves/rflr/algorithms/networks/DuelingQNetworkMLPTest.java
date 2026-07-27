package br.com.guialves.rflr.algorithms.networks;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
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

    private static final float DELTA = 1e-6f;
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
    @EnumSource(DuelingType.class)
    void shouldAvoidMemoryLeak(DuelingType duelingType) {
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

    /**
     * Verifies the Dueling DQN mean decomposition:
     *
     * <pre>
     * Q(s, a) = V(s) + (A(s, a) - mean(A))
     *
     * mean(Q) = V
     *
     * Q - mean(Q) = A - mean(A)
     *
     * mean(A - mean(A)) = 0
     * </pre>
     *
     * Therefore, the recovered normalized advantage must have zero mean across
     * the action dimension.
     */
    @Test
    void shouldCheckIfAdvantageHasZeroMean() {
        int obs = 8;
        int actions = 4;
        @Cleanup var net = DuelingQNetworkMLP.withMeanType(obs, actions, manager);
        var input = manager.randomUniform(0f, 1f, new Shape(64, obs));
        var q = net.forward(input);
        var meanQ = q.mean(new int[] {1}, true);
        var recoveredAdvantage = q.sub(meanQ);
        var advMean = recoveredAdvantage.mean(new int[] {1});
        assertTrue(advMean.abs().max().getFloat() < DELTA);
    }

    /**
     * Verifies the Dueling DQN max decomposition:
     *
     * <pre>
     * Q(s, a) = V(s) + (A(s, a) - max(A))
     *
     * max(Q) = V
     *
     * Q - max(Q) = A - max(A)
     *
     * max(A - max(A)) = 0
     * </pre>
     *
     * Therefore, the recovered normalized advantage must have zero maximum across
     * the action dimension.
     */
    @Test
    void shouldCheckIfAdvantageHasZeroMax() {
        int obs = 8;
        int actions = 4;
        @Cleanup var net = DuelingQNetworkMLP.withMaxType(obs, actions, manager);
        var input = manager.randomUniform(0f, 1f, new Shape(64, obs));
        var q = net.forward(input);
        var maxQ = q.max(new int[] {1}, true);
        var recoveredAdvantage = q.sub(maxQ);
        var advMax = recoveredAdvantage.max(new int[] {1});
        assertTrue(advMax.abs().max().getFloat() < DELTA);
    }
}
