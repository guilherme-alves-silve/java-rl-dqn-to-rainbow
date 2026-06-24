package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AgentDQNTest {

    private static final float DELTA = 0.001f;

    @Mock
    private IEnv env;
    @Mock
    private Device device;
    @Mock
    private Optimizer optimizer;
    @Mock
    private IDeepQNetwork onlineNet;
    @Mock
    private IDeepQNetwork targetNet;
    @Mock
    private Block mockBlock;
    @Mock
    private PlotTrackers plotTrackers;
    @Mock
    private NDArray mockState;
    @Mock
    private ActionSpaceType.ActionResult mockActionResult;

    private AgentDQN agent;

    private final int updateQTargetAtTimeN = 10;
    private final float minEpsilon = 0.1f;
    private final float epsilonDecay = 0.9f;
    private final float gamma = 0.99f;

    @BeforeEach
    void setUp() {
        lenient().when(onlineNet.clone()).thenReturn(targetNet);
        lenient().when(onlineNet.getBlock()).thenReturn(mockBlock);
        lenient().when(targetNet.getBlock()).thenReturn(mockBlock);
        lenient().when(mockBlock.getParameters()).thenReturn(new ParameterList());
        lenient().when(env.actionSpaceType()).thenReturn(ActionSpaceType.DISCRETE);

        Supplier<IDeepQNetwork> networkFactory = () -> onlineNet;

        float initialEpsilon = 1.0f;
        agent = new AgentDQN(
                initialEpsilon,
                updateQTargetAtTimeN,
                minEpsilon,
                epsilonDecay,
                gamma,
                env,
                device,
                optimizer,
                networkFactory,
                plotTrackers
        );
    }

    @Test
    @DisplayName("Should decay epsilon correctly while respecting the minimum threshold")
    void shouldDecayEpsilonCorrectlyAndRespectMinimum() {
        float newEpsilon = agent.reduceEpsilon(1.0f);
        assertEquals(0.9f, newEpsilon, DELTA);

        float epsilonAtFloor = agent.reduceEpsilon(0.105f);
        assertEquals(0.1f, epsilonAtFloor, DELTA);
    }

    @Test
    @DisplayName("Should select a random action when random value is less than epsilon (Exploration)")
    void shouldSelectRandomActionWhenExploring() {
        when(env.actionSpaceSample()).thenReturn(mockActionResult);

        ActionSpaceType.ActionResult result = agent.selectAction(mockState);

        assertNotNull(result);
        verify(env, times(1)).actionSpaceSample();
        verify(onlineNet, never()).forward(any(NDArray.class));
    }

    @Test
    @DisplayName("Should select greedy action from online network when epsilon is zero (Exploitation)")
    void shouldSelectGreedyActionFromNetworkWhenExploiting() {
        AgentDQN exploitativeAgent = new AgentDQN(
                0.0f,
                updateQTargetAtTimeN, minEpsilon, epsilonDecay, gamma,
                env, device, optimizer, () -> onlineNet, plotTrackers
        );

        NDArray mockExpandedState = mock(NDArray.class);
        NDArray mockForwardOutput = mock(NDArray.class);
        NDArray mockStopGrad = mock(NDArray.class);
        NDArray mockArgMax = mock(NDArray.class);

        when(mockState.expandDims(0)).thenReturn(mockExpandedState);
        when(onlineNet.forward(mockExpandedState)).thenReturn(mockForwardOutput);
        when(mockForwardOutput.stopGradient()).thenReturn(mockStopGrad);
        when(mockStopGrad.argMax(1)).thenReturn(mockArgMax);
        when(mockArgMax.getLong(0)).thenReturn(2L);

        ActionSpaceType.ActionResult result = exploitativeAgent.selectAction(mockState);

        assertNotNull(result);
        verify(onlineNet, times(1)).forward(any(NDArray.class));
        verify(mockArgMax, times(1)).close();
    }

    @Test
    @DisplayName("Should throw NullPointerException when training parameters are null")
    void shouldThrowExceptionWhenTrainParametersAreNull() {
        assertThrows(NullPointerException.class, () -> {
            agent.train(32, 100, null, null);
        });
    }
}
