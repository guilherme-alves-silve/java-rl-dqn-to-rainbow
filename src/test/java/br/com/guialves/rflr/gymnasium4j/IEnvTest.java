package br.com.guialves.rflr.gymnasium4j;

import br.com.guialves.rflr.fixture.MockDiscreteEnv;
import lombok.Cleanup;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests of the {@link IEnv} contract.
 *
 * <p>The focus is {@link IEnv#throwIfNotSupported(int)}, which validates the number of actions of
 * the configuration against the action space of the environment, and the default methods that only
 * make sense for a vectorized environment ({@code stepVec} and {@code renderVec}).</p>
 *
 * <p>The environment works with the <b>batched</b> action space of a vectorized environment: the
 * type is {@code MULTI_DISCRETE} and the space is {@code MultiDiscrete([n, n, ...])}, one entry per
 * environment. The number of actions of the configuration describes a <b>single</b> environment, so
 * every entry of the batched space must have the configured number of actions.</p>
 */
class IEnvTest {

    @Nested
    @DisplayName("throwIfNotSupported  DISCRETE action space")
    class DiscreteActionSpaceTests {

        @Test
        @DisplayName("Should accept a discrete environment with the configured number of actions")
        void shouldAcceptADiscreteEnvironmentWithTheConfiguredNumberOfActions() {
            @Cleanup var env = new MockDiscreteEnv(4, 6);

            assertDoesNotThrow(() -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should reject a discrete environment with a different number of actions")
        void shouldRejectADiscreteEnvironmentWithADifferentNumberOfActions() {
            @Cleanup var env = new MockDiscreteEnv(4, 4);

            var exception = assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(6));
            assertTrue(exception.getMessage().contains("Incorrect number of actions"),
                    "the error must explain the mismatch: " + exception.getMessage());
            assertTrue(exception.getMessage().contains("Discrete(6)"),
                    "the error must show the configured space: " + exception.getMessage());
            assertTrue(exception.getMessage().contains("Discrete(4)"),
                    "the error must show the space of the environment: " + exception.getMessage());
        }

        @Test
        @DisplayName("Should validate the discrete space against the space of a single environment")
        void shouldValidateTheDiscreteSpaceAgainstTheSingleEnvironmentSpace() {
            @Cleanup var env = mockEnv(ActionSpaceType.DISCRETE, true,
                    "MultiDiscrete([6 6 6 6])", "Discrete(6)");
            assertDoesNotThrow(() -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should reject when the space of a single environment does not match")
        void shouldRejectWhenTheSingleEnvironmentSpaceDoesNotMatch() {
            @Cleanup var env = mockEnv(ActionSpaceType.DISCRETE, true,
                    "MultiDiscrete([4 4 4 4])", "Discrete(4)");

            assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(6));
        }
    }

    @Nested
    @DisplayName("throwIfNotSupported  MULTI_DISCRETE action space")
    class MultiDiscreteActionSpaceTests {

        @Test
        @DisplayName("Should accept a batched space with the configured actions in every environment")
        void shouldAcceptABatchedSpaceWithTheConfiguredActions() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([6 6 6 6])");
            assertDoesNotThrow(() -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should accept a batched space with a single environment")
        void shouldAcceptABatchedSpaceWithASingleEnvironment() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([6])");
            assertDoesNotThrow(() -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should accept a batched space with many environments")
        void shouldAcceptABatchedSpaceWithManyEnvironments() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2])");
            assertDoesNotThrow(() -> env.throwIfNotSupported(2));
        }

        @Test
        @DisplayName("Should accept a batched space with a single action")
        void shouldAcceptABatchedSpaceWithASingleAction() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([1 1 1])");
            assertDoesNotThrow(() -> env.throwIfNotSupported(1));
        }

        @Test
        @DisplayName("Should reject when one environment has a different number of actions")
        void shouldRejectWhenOneEnvironmentHasADifferentNumberOfActions() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([6 6 5 6])");
            var exception = assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(6));
            assertTrue(exception.getMessage().contains("Incorrect number of actions"),
                    "the error must explain the mismatch: " + exception.getMessage());
            assertTrue(exception.getMessage().contains("MultiDiscrete([6 6 5 6])"),
                    "the error must show the space of the environment: " + exception.getMessage());
        }

        @Test
        @DisplayName("Should reject when the number of actions is different")
        void shouldRejectWhenTheNumberOfActionsIsDifferent() {
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([6 6 6 6])");
            assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(4));
        }

        @Test
        @DisplayName("Should reject actions that only share the prefix of the configured one")
        void shouldRejectActionsThatOnlyShareThePrefix() {
            // 16 actions must not be accepted as 6 actions
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([16 16])");
            assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should reject a space that is not a MultiDiscrete")
        void shouldRejectASpaceThatIsNotAMultiDiscrete() {
            @Cleanup var env = mockVectorizedEnv("Discrete(6)");
            assertThrows(IllegalArgumentException.class, () -> env.throwIfNotSupported(6));
        }

        @Test
        @DisplayName("Should reject a MultiDiscrete environment that is not vectorized")
        void shouldRejectAMultiDiscreteThatIsNotVectorized() {
            @Cleanup var env = mockEnv(ActionSpaceType.MULTI_DISCRETE, false,
                    "MultiDiscrete([2 3])", "MultiDiscrete([2 3])");
            var exception = assertThrows(UnsupportedOperationException.class,
                    () -> env.throwIfNotSupported(6));
            assertTrue(exception.getMessage().contains("must be vectorized"),
                    "the error must explain that the environment has to be vectorized: " + 
                    exception.getMessage());
        }

        @Test
        @DisplayName("Known limitation: the pattern does not require a separator between the actions")
        void shouldDocumentThatThePatternDoesNotRequireASeparator() {
            // The pattern is MultiDiscrete\((\[6\s*)]\), so "MultiDiscrete([66])" is accepted for
            // 6 actions. It never happens with gymnasium (the space is always space separated), but
            // it makes the validation looser than it looks.
            // TODO: Correct this
            @Cleanup var env = mockVectorizedEnv("MultiDiscrete([66])");

            assertDoesNotThrow(() -> env.throwIfNotSupported(6));
        }
    }

    @Nested
    @DisplayName("throwIfNotSupported  unsupported action spaces")
    class UnsupportedActionSpaceTests {

        @ParameterizedTest(name = "[{index}] {0}")
        @EnumSource(value = ActionSpaceType.class, names = {"BOX", "TEXT", "MULTI_BINARY", "UNKNOWN"})
        @DisplayName("Should reject the action spaces that are not supported yet")
        void shouldRejectTheUnsupportedActionSpaces(ActionSpaceType spaceType) {
            @Cleanup var env = mockEnv(spaceType, false, "Box(0, 1, (3,), float32)", "Box(0, 1, (3,), float32)");

            var exception = assertThrows(UnsupportedOperationException.class,
                    () -> env.throwIfNotSupported(6));
            assertTrue(exception.getMessage().contains("not supported yet"),
                    "the error must explain that the space is not supported: " + exception.getMessage());
        }

        @Test
        @DisplayName("Should fail when the environment does not report its action space type")
        void shouldFailWhenTheActionSpaceTypeIsNull() {
            @Cleanup var env = mockEnv(null, false, "Discrete(6)", "Discrete(6)");

            // the switch has no "case null", so the type is dereferenced before the switch
            assertThrows(NullPointerException.class, () -> env.throwIfNotSupported(6));
        }
    }

    @Nested
    @DisplayName("Default methods")
    class DefaultMethodsTests {

        @Test
        @DisplayName("Should refuse to step a environment that is not vectorized")
        void shouldRefuseToStepANonVectorizedEnvironment() {
            @Cleanup var env = mock(IEnv.class, CALLS_REAL_METHODS);

            var exception = assertThrows(IllegalArgumentException.class,
                    () -> env.stepVec(null, null));
            assertTrue(exception.getMessage().contains("stepVec"),
                    "the error must name the method: " + exception.getMessage());
        }

        @Test
        @DisplayName("Should refuse to render a environment that is not vectorized")
        void shouldRefuseToRenderANonVectorizedEnvironment() {
            @Cleanup var env = mock(IEnv.class, CALLS_REAL_METHODS);

            var exception = assertThrows(IllegalArgumentException.class, env::renderVec);
            assertTrue(exception.getMessage().contains("renderVec"),
                    "the error must name the method: " + exception.getMessage());
        }

        @Test
        @DisplayName("Should expose the action space of the environment as the space of a single one")
        void shouldExposeTheActionSpaceAsTheSingleEnvironmentSpace() {
            @Cleanup var env = mock(IEnv.class, CALLS_REAL_METHODS);
            when(env.actionSpaceStr()).thenReturn("Discrete(6)");

            assertEquals("Discrete(6)", env.singleActionSpaceStr());
        }
    }

    private IEnv mockVectorizedEnv(String actionSpaceStr) {
        return mockEnv(ActionSpaceType.MULTI_DISCRETE, true, actionSpaceStr, actionSpaceStr);
    }

    /**
     * Builds an environment with the given action space, running the default methods of
     * {@link IEnv} for real.
     */
    private IEnv mockEnv(ActionSpaceType spaceType,
                         boolean vectorized,
                         String actionSpaceStr,
                         String singleActionSpaceStr) {
        var env = mock(IEnv.class, CALLS_REAL_METHODS);
        when(env.actionSpaceType()).thenReturn(spaceType);
        when(env.vectorized()).thenReturn(vectorized);
        when(env.actionSpaceStr()).thenReturn(actionSpaceStr);
        when(env.singleActionSpaceStr()).thenReturn(singleActionSpaceStr);
        when(env.envName()).thenReturn("TestEnv");
        return env;
    }
}
