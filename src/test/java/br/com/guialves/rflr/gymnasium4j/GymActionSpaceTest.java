package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.*;
import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@DisplayName("Gym Environment Action Space Tests")
class GymActionSpaceTest {

    private NDManager ndManager;

    @BeforeEach
    void setUp() {
        ndManager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        if (ndManager != null) {
            ndManager.close();
        }
    }

    @Nested
    @DisplayName("Discrete Action Space Environments")
    class DiscreteActionSpaceTests {

        @Test
        @DisplayName("CartPole-v1 should have Discrete(2) action space")
        void testCartPoleActionSpace() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                log.info("CartPole action space: {}", env.actionSpaceStr());

                // Verify action space string
                assertTrue(env.actionSpaceStr().contains("Discrete(2)"));

                // Sample and verify action type
                try (var action = env.actionSpaceSample()) {
                    assertEquals(DISCRETE, action.spaceType());

                    // Extract and verify value
                    Long value = action.value();
                    assertNotNull(value);
                    assertTrue(value == 0 || value == 1, "CartPole action should be 0 or 1");

                    log.info("Sampled action value: {}", value);
                }

                // Test with manual actions
                try (var action0 = DISCRETE.get(0)) {
                    assertEquals(DISCRETE, action0.spaceType());
                    assertEquals(0L, action0.<Long>value());
                }

                try (var action1 = DISCRETE.get(1)) {
                    assertEquals(DISCRETE, action1.spaceType());
                    assertEquals(1L, action1.<Long>value());
                }
            }
        }

        @Test
        @DisplayName("MountainCar-v0 should have Discrete(3) action space")
        void testMountainCarActionSpace() {
            try (var env = Gym.make("MountainCar-v0", ndManager)) {
                log.info("MountainCar action space: {}", env.actionSpaceStr());

                assertTrue(env.actionSpaceStr().contains("Discrete(3)"));

                try (var action = env.actionSpaceSample()) {
                    assertEquals(DISCRETE, action.spaceType());

                    Long value = action.value();
                    assertNotNull(value);
                    assertTrue(value >= 0 && value <= 2, "MountainCar action should be 0, 1, or 2");

                    log.info("Sampled action value: {}", value);
                }
            }
        }

        @Test
        @DisplayName("Acrobot-v1 should have Discrete(3) action space")
        void testAcrobotActionSpace() {
            try (var env = Gym.make("Acrobot-v1", ndManager)) {
                log.info("Acrobot action space: {}", env.actionSpaceStr());

                assertTrue(env.actionSpaceStr().contains("Discrete(3)"));

                try (var action = env.actionSpaceSample()) {
                    assertEquals(DISCRETE, action.spaceType());

                    Long value = action.value();
                    assertNotNull(value);
                    assertTrue(value >= 0 && value <= 2);
                }
            }
        }
    }

    @Nested
    @DisplayName("Box (Continuous) Action Space Environments")
    class BoxActionSpaceTests {

        @Test
        @DisplayName("Pendulum-v1 should have Box(1,) action space")
        void testPendulumActionSpace() {
            try (var env = Gym.make("Pendulum-v1", ndManager)) {
                log.info("Pendulum action space: {}", env.actionSpaceStr());

                assertTrue(env.actionSpaceStr().contains("Box"));

                try (var action = env.actionSpaceSample()) {
                    assertEquals(BOX, action.spaceType());

                    // Pendulum has 1D continuous action
                    Object value = action.value();
                    assertNotNull(value);

                    // Could be Double (scalar) or double[] (array)
                    if (value instanceof Double d) {
                        assertTrue(d >= -2.0 && d <= 2.0, "Pendulum action should be in [-2, 2]");
                        log.info("Sampled action (scalar): {}", d);
                    } else if (value instanceof double[] arr) {
                        assertEquals(1, arr.length);
                        assertTrue(arr[0] >= -2.0 && arr[0] <= 2.0);
                        log.info("Sampled action (array): {}", arr[0]);
                    } else {
                        fail("Expected Double or double[] but got " + value.getClass());
                    }
                }

                try (var action = BOX.get(new double[]{1.5})) {
                    assertEquals(BOX, action.spaceType());
                    double[] value = action.value();
                    assertEquals(1, value.length);
                    assertEquals(1.5, value[0], 0.001);
                }
            }
        }

        @Test
        @DisplayName("MountainCarContinuous-v0 should have Box(1,) action space")
        void testMountainCarContinuousActionSpace() {
            try (var env = Gym.make("MountainCarContinuous-v0", ndManager)) {
                log.info("MountainCarContinuous action space: {}", env.actionSpaceStr());

                assertTrue(env.actionSpaceStr().contains("Box"));

                try (var action = env.actionSpaceSample()) {
                    assertEquals(BOX, action.spaceType());

                    Object value = action.value();
                    assertNotNull(value);

                    if (value instanceof Double d) {
                        assertTrue(d >= -1.0 && d <= 1.0);
                        log.info("Sampled action: {}", d);
                    } else if (value instanceof double[] arr) {
                        assertEquals(1, arr.length);
                        assertTrue(arr[0] >= -1.0 && arr[0] <= 1.0);
                        log.info("Sampled action: {}", arr[0]);
                    }
                }
            }
        }

        @Test
        @DisplayName("LunarLanderContinuous-v3 should have Box(2,) action space")
        void testLunarLanderContinuousActionSpace() {
            try (var env = Gym.make("LunarLanderContinuous-v3", ndManager)) {
                log.info("LunarLanderContinuous action space: {}", env.actionSpaceStr());

                assertTrue(env.actionSpaceStr().contains("Box"));

                try (var action = env.actionSpaceSample()) {
                    assertEquals(BOX, action.spaceType());

                    double[] value = action.value();
                    assertNotNull(value);
                    assertEquals(2, value.length);

                    assertTrue(value[0] >= -1.0 && value[0] <= 1.0);
                    assertTrue(value[1] >= -1.0 && value[1] <= 1.0);

                    log.info("Sampled action: [{}, {}]", value[0], value[1]);
                }

                try (var action = BOX.get(new double[]{0.5, -0.3})) {
                    assertEquals(BOX, action.spaceType());
                    double[] value = action.value();
                    assertArrayEquals(new double[]{0.5, -0.3}, value, 0.001);
                }
            }
        }
    }

    @Nested
    @DisplayName("Action Space Integration Tests")
    class ActionSpaceIntegrationTests {

        @Test
        @DisplayName("Should correctly use discrete actions in CartPole")
        void testDiscreteActionInEnvironment() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                env.reset();

                try (var action = DISCRETE.get(0)) {
                    var result = env.step(action);
                    assertNotNull(result);
                    assertNotNull(result.state());

                    log.info("Step with action 0 - reward: {}, done: {}",
                            result.reward(), result.done());
                }

                try (var action = DISCRETE.get(1)) {
                    var result = env.step(action);
                    assertNotNull(result);
                    assertNotNull(result.state());

                    log.info("Step with action 1 - reward: {}, done: {}",
                            result.reward(), result.done());
                }
            }
        }

        @Test
        @DisplayName("Should correctly use continuous actions in Pendulum")
        void testContinuousActionInEnvironment() {
            try (var env = Gym.make("Pendulum-v1", ndManager)) {
                env.reset();

                double[] testActions = {-2.0, -1.0, 0.0, 1.0, 2.0};

                for (double actionValue : testActions) {
                    try (var action = BOX.get(new double[]{actionValue})) {
                        var result = env.step(action);
                        assertNotNull(result);
                        assertNotNull(result.state());

                        log.info("Step with action {} - reward: {}",
                                actionValue, result.reward());
                    }

                    env.reset();
                }
            }
        }

        @Test
        @DisplayName("Should handle sampled actions correctly")
        void testSampledActionInEnvironment() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                env.reset();

                for (int i = 0; i < 10; i++) {
                    try (var action = env.actionSpaceSample()) {
                        assertEquals(DISCRETE, action.spaceType());

                        Long actionValue = action.value();
                        assertNotNull(actionValue);
                        assertTrue(actionValue >= 0 && actionValue <= 1);

                        var result = env.step(action);
                        assertNotNull(result);

                        log.info("Step {} - action: {}, reward: {}, done: {}",
                                i, actionValue, result.reward(), result.done());

                        if (result.done()) {
                            break;
                        }
                    }
                }
            }
        }

        @Test
        @DisplayName("Should preserve action value through step")
        void testActionValuePreservation() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                env.reset();

                int expectedAction = 1;
                try (var action = DISCRETE.get(expectedAction)) {

                    Long valueBefore = action.value();
                    assertEquals(expectedAction, valueBefore);

                    var result = env.step(action);
                    assertNotNull(result);

                    Long valueAfter = action.value();
                    assertEquals(expectedAction, valueAfter);
                    assertEquals(valueBefore, valueAfter);
                }
            }
        }

        @Test
        @DisplayName("Should handle multiple action types in sequence")
        void testMultipleActionTypes() {

            try (var env1 = Gym.make("CartPole-v1", ndManager)) {
                env1.reset();

                try (var action = DISCRETE.get(0)) {
                    assertEquals(DISCRETE, action.spaceType());
                    assertEquals(0L, action.<Long>value());
                    env1.step(action);
                }
            }

            try (var env2 = Gym.make("Pendulum-v1", ndManager)) {
                env2.reset();

                try (var action = BOX.get(new double[]{1.0})) {
                    assertEquals(BOX, action.spaceType());
                    double[] value = action.value();
                    assertEquals(1.0, value[0], 0.001);
                    env2.step(action);
                }
            }
        }
    }

    @Nested
    @DisplayName("Action Space Edge Cases")
    class ActionSpaceEdgeCaseTests {

        @Test
        @DisplayName("Should handle extreme discrete action values")
        void testExtremeDiscreteActions() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                env.reset();

                // Test boundary values
                try (var action = DISCRETE.get(0)) {
                    assertEquals(0L, action.<Long>value());
                    var result = env.step(action);
                    assertNotNull(result);
                }

                try (var action = DISCRETE.get(1)) {
                    assertEquals(1L, action.<Long>value());
                    var result = env.step(action);
                    assertNotNull(result);
                }
            }
        }

        @Test
        @DisplayName("Should handle extreme continuous action values")
        void testExtremeContinuousActions() {
            try (var env = Gym.make("Pendulum-v1", ndManager)) {
                env.reset();

                try (var action = BOX.get(new double[]{-2.0})) {
                    double[] value = action.value();
                    assertEquals(-2.0, value[0], 0.001);
                    var result = env.step(action);
                    assertNotNull(result);
                }

                env.reset();

                try (var action = BOX.get(new double[]{2.0})) {
                    double[] value = action.value();
                    assertEquals(2.0, value[0], 0.001);
                    var result = env.step(action);
                    assertNotNull(result);
                }
            }
        }

        @Test
        @DisplayName("Should handle zero actions")
        void testZeroActions() {

            try (var env = Gym.make("MountainCar-v0", ndManager)) {
                env.reset();

                try (var action = DISCRETE.get(0)) {
                    assertEquals(0L, action.<Long>value());
                    var result = env.step(action);
                    assertNotNull(result);
                }
            }

            // Continuous zero
            try (var env = Gym.make("Pendulum-v1", ndManager)) {
                env.reset();

                try (var action = BOX.get(new double[]{0.0})) {
                    double[] value = action.value();
                    assertEquals(0.0, value[0], 0.001);
                    var result = env.step(action);
                    assertNotNull(result);
                }
            }
        }
    }

    @Nested
    @DisplayName("Action Space Type Safety Tests")
    class ActionSpaceTypeSafetyTests {

        @Test
        @DisplayName("Should enforce correct action type for discrete environments")
        void testDiscreteActionTypeSafety() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                try (var action = env.actionSpaceSample()) {
                    // Should be discrete
                    assertEquals(DISCRETE, action.spaceType());

                    // Should extract as Long
                    Long value = action.valueAs(Long.class);
                    assertNotNull(value);

                    // Should fail to extract as wrong type
                    assertThrows(ClassCastException.class,
                            () -> action.valueAs(Double.class));
                }
            }
        }

        @Test
        @DisplayName("Should enforce correct action type for continuous environments")
        void testContinuousActionTypeSafety() {
            try (var env = Gym.make("Pendulum-v1", ndManager)) {
                try (var action = env.actionSpaceSample()) {
                    // Should be box
                    assertEquals(BOX, action.spaceType());

                    // Value should be either Double or double[]
                    Object value = action.value();
                    assertTrue(value instanceof Double || value instanceof double[],
                            "Expected Double or double[] but got " + value.getClass());
                }
            }
        }

        @Test
        @DisplayName("Should provide consistent value on multiple calls")
        void testValueConsistency() {
            try (var env = Gym.make("CartPole-v1", ndManager)) {
                try (var action = DISCRETE.get(1)) {
                    Long value1 = action.value();
                    Long value2 = action.value();
                    Long value3 = action.value();

                    assertEquals(value1, value2);
                    assertEquals(value2, value3);
                    assertEquals(1L, value1);
                }
            }
        }
    }
}
