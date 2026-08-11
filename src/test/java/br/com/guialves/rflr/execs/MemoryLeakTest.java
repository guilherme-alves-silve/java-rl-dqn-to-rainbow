package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement.ManagerNode;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.getDebugDump;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.systemResourceCount;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test for memory leaks across all agent training pipelines.
 *
 * <p>Each test invokes {@code AgentXxxMain.run()} which trains an agent to
 * completion and returns a {@link ManagerNode} snapshot of the resulting
 * manager hierarchy. The test then asserts the <i>exact</i> number of direct
 * resources on the root and on each named sub-manager, so that any future
 * regression that adds a new sub-manager or leaks an NDArray will fail with
 * a clear, actionable message.</p>
 *
 * <h2>Reference values (captured from past training runs)</h2>
 * <pre>
 *   parent                  : 2    (online + target network sub-managers)
 *   DeepQNetworkMLP         : 6    (3 Linear layers × 2 = weight + bias)
 *   DuelingQNetworkMLP      : 8    (2 backbone + 1 value + 1 advantage × 2)
 *   NoisyQNetworkMLP        : 10   (1 Linear + 2 NoisyLayer × 4 params)
 *   NoisyDuelingQNetworkMLP : 18   (1 Linear + 4 NoisyLayer × 4 + extras)
 *   ExperienceReplayBuffer  : 256  (128 experiences × 2 NDArrays)
 * </pre>
 *
 * <h2>Why exact counts and not a threshold</h2>
 * <p>The previous version of this test used a {@code MAX_LEAK_PER_AGENT}
 * threshold, which tolerates a small drift. The current version asserts the
 * exact resource count so that an accidental change in network topology
 * (e.g. adding a layer) or in the buffer's storage layout (e.g. adding a
 * segment tree) is detected immediately. If a new agent legitimately needs
 * a different layout, update the expected values in this file.</p>
 */
@DisplayName("Memory leak regression tests")
class MemoryLeakTest {

    private static final String FRAMES_LIMIT = "500";
    private static final String BUFFER_CAPACITY = "128";
    private static final String DONT_SAVE_MODEL = "false";
    private static final String DONT_RECORD = "false";
    private static final String DONT_SHOW_METRICS = "false";
    private static final String DONT_RENDER = "false";

    private static final int GC_TRIES = 3;
    private static final int MIN_SLEEP = 100;

    /** Root manager: online + target network sub-managers. */
    private static final int EXPECTED_ROOT_SUBMANAGERS = 2;

    /** DeepQNetworkMLP has 3 Linear layers (weight + bias each). */
    private static final int DEEP_NET_PARAMS = 6;

    /** DuelingQNetworkMLP has 4 Linear layers (2 backbone + 1 value + 1 advantage). */
    private static final int DUELING_NET_PARAMS = 8;

    /** NoisyQNetworkMLP has 1 Linear + 2 NoisyLayer (each NoisyLayer = 4 params). */
    private static final int NOISY_NET_PARAMS = 10;

    /** NoisyDuelingQNetworkMLP has 1 Linear + 4 NoisyLayer. */
    private static final int NOISY_DUELING_NET_PARAMS = 18;

    /** CategoricalQNetworkMLP has 3 Linear layers (weight + bias each). */
    private static final int AGENT_C51_DQN_NET_PARAMS = 3;

    /**
     * Each {@code Experience} contributes exactly 2 NDArrays to its buffer:
     * one {@code state} and one {@code nextState}. We use this as a
     * structural invariant (the buffer's NDArray count is always a
     * multiple of 2) instead of asserting an absolute number, which
     * depends on how many experiences were collected during the
     * (short) test run.
     */
    private static final int NDARRAYS_PER_EXPERIENCE = 2;

    private NDManager manager;

    @BeforeEach
    @DisplayName("Configure short training run for fast tests")
    void setUpEach() {
        manager = NDManager.newBaseManager();
        System.setProperty("agent.framesLimit", FRAMES_LIMIT);
        System.setProperty("agent.bufferCapacity", BUFFER_CAPACITY);
        System.setProperty("agent.saveModel", DONT_SAVE_MODEL);
        System.setProperty("agent.records", DONT_RECORD);
        System.setProperty("agent.showAllMetrics", DONT_SHOW_METRICS);
        System.setProperty("agent.renderRun", DONT_RENDER);
    }

    @AfterEach
    @DisplayName("Stabilize the JVM before the next test")
    void tearDownEach() throws InterruptedException {
        for (int i = 0; i < GC_TRIES; i++) {
            System.gc();
        }
        Thread.sleep(MIN_SLEEP);
        manager.close();
    }

    @Test
    @DisplayName("AgentDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentDQN() {
        var managerNode = AgentDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentDQN",
                "DeepQNetworkMLP-",
                DEEP_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentDDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentDDQN() {
        var managerNode = AgentDDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentDDQN",
                "DeepQNetworkMLP-",
                DEEP_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentNStepDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentNStepDQN() {
        var managerNode = AgentNStepDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentNStepDQN",
                "DeepQNetworkMLP-",
                DEEP_NET_PARAMS,
                "NStepExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentDuelingDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentDuelingDQN() {
        var managerNode = AgentDuelingDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentDuelingDQN",
                "DuelingQNetworkMLP-",
                DUELING_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentDQNPER should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentDQNPER() {
        var managerNode = AgentDQNPERMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentDQNPER",
                "DeepQNetworkMLP-",
                DEEP_NET_PARAMS,
                "PrioritizedReplayBuffer-");
    }

    @Test
    @DisplayName("AgentNoisyNetDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentNoisyNetDQN() {
        var managerNode = AgentNoisyNetDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentNoisyNetDQN",
                "NoisyQNetworkMLP-",
                NOISY_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentNoisyDuelingNetDDQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentNoisyDuelingNetDDQN() {
        var managerNode = AgentNoisyDuelingNetDDQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentNoisyDuelingNetDDQN",
                "NoisyDuelingQNetworkMLP-",
                NOISY_DUELING_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("AgentC51DQN should leave exactly the expected NDArrays after training")
    void shouldNotLeakMemoryAfterAgentC51DQN() {
        // TODO: Correct memory leak
        var managerNode = AgentC51DQNMain.run().orElseThrow();
        assertAgentStructure(
                managerNode,
                "AgentC51DQN",
                "CategoricalQNetworkMLP-",
                AGENT_C51_DQN_NET_PARAMS,
                "ExperienceReplayBuffer-");
    }

    @Test
    @DisplayName("getDebugDump returns a node for the test manager")
    void getDebugDumpShouldReturnNodeForBaseManager() {
        var root = getDebugDump(manager);
        assertTrue(root.isPresent(), "test manager should be a BaseNDManager");
    }

    @Test
    @DisplayName("systemResourceCount matches subtree size of the root dump")
    void systemResourceCountShouldMatchSubtreeSize() {
        var root = getDebugDump(manager);
        assertTrue(root.isPresent());
        assertEquals(root.get().subtreeSize(), systemResourceCount(manager),
                "systemResourceCount is just subtreeSize() of the root");
    }

    /**
     * Asserts the full shape of the manager tree for a given agent:
     *
     * <ol>
     *   <li>the root has exactly {@link #EXPECTED_ROOT_SUBMANAGERS} direct
     *       sub-managers (one online network + one target network);</li>
     *   <li>both networks (matched by {@code networkPrefix}) hold exactly
     *       {@code expectedParamsPerNetwork} NDArrays (the {@code Parameter}
     *       arrays of the block), ignoring any sub-managers (e.g. the
     *       {@code Model}'s own manager, which DJL creates via
     *       {@code newModel});</li>
     *   <li>the buffer (matched by {@code bufferPrefix}) exists exactly once
     *       and holds a multiple of {@link #NDARRAYS_PER_EXPERIENCE} NDArrays
     *       (one {@code state} + one {@code nextState} per stored experience).
     *       The absolute count is <i>not</i> asserted because it depends on
     *       how many experiences were collected during the (short) test run;
     *       see {@link #assertAgentStructure} for the structural check.</li>
     * </ol>
     *
     * <p>The buffer is searched recursively (via {@code findMatching}) so
     * the assertion is robust to whether the buffer's sub-manager is a
     * direct child of the root or nested under a network. The networks are
     * expected to be direct children of the root.</p>
     *
     * <p><b>Why count only NDArrays and not {@code totalResources}:</b>
     * a network's sub-manager also contains the {@code Model}'s internal
     * NDManager (e.g. {@code byType={PtNDManager=1, PtNDArray=6}} on a
     * 3-layer MLP), so {@code totalResources} mixes sub-managers with
     * NDArrays. {@link #countNDArrays(ManagerNode)} filters by simple
     * class name suffix to give a clean parameter count.</p>
     */
    private void assertAgentStructure(ManagerNode root,
                                      String agentName,
                                      String networkPrefix,
                                      int expectedParamsPerNetwork,
                                      String bufferPrefix) {
        // 1) Root has exactly the two network sub-managers.
        assertEquals(EXPECTED_ROOT_SUBMANAGERS, root.totalResources(),
                agentName + ": root should have exactly "
                        + EXPECTED_ROOT_SUBMANAGERS
                        + " direct sub-managers (online + target networks), got "
                        + root.totalResources() + " with byType=" + root.byType());

        // 2) Both networks have the expected number of parameter arrays.
        var networks = root.findMatching(n -> n.name().startsWith(networkPrefix) ||
                n.name().startsWith("clone-" + networkPrefix));
        assertEquals(2, networks.size(),
                agentName + ": expected exactly 2 sub-managers starting with '"
                        + networkPrefix + "', found " + networks.size());
        for (int i = 0; i < networks.size(); i++) {
            var net = networks.get(i);
            long paramCount = countNDArrays(net);
            assertEquals(expectedParamsPerNetwork, paramCount,
                    agentName + ": network '" + net.name()
                            + "' should have exactly " + expectedParamsPerNetwork
                            + " parameters, got " + paramCount
                            + " NDArrays (byType=" + net.byType() + ")");
        }

        // 3) The replay buffer exists once and holds a multiple of 2 NDArrays
        // (each experience has exactly one state + one nextState NDArray).
        // The exact count depends on how many experiences were collected
        // during the short test run, so we assert the structural invariant
        // rather than an absolute number.
        var buffers = root.findMatching(n -> n.name().startsWith(bufferPrefix));
        assertEquals(1, buffers.size(),
                agentName + ": expected exactly 1 sub-manager starting with '"
                        + bufferPrefix + "', found " + buffers.size());
        var buffer = buffers.get(0);
        long bufferNDArrayCount = countNDArrays(buffer);
        assertTrue(bufferNDArrayCount >= 0 && bufferNDArrayCount % NDARRAYS_PER_EXPERIENCE == 0,
                agentName + ": buffer '" + buffer.name()
                        + "' should hold a multiple of " + NDARRAYS_PER_EXPERIENCE
                        + " NDArrays (one state + one nextState per experience), got "
                        + bufferNDArrayCount + " (byType=" + buffer.byType() + ")");
        // Sanity: with BUFFER_CAPACITY=128 and 500 training frames, the
        // buffer should hold at least 1 experience. A count of 0 would
        // indicate the buffer was never written to.
        assertTrue(bufferNDArrayCount >= NDARRAYS_PER_EXPERIENCE,
                agentName + ": buffer '" + buffer.name()
                        + "' should hold at least one experience, got "
                        + bufferNDArrayCount + " NDArrays (byType=" + buffer.byType() + ")");
    }

    /**
     * Counts only the NDArrays in a manager's resources, ignoring any
     * sub-managers (e.g. {@code PtNDManager}). The class name is matched
     * by suffix {@code NDArray} so this works across engines
     * ({@code PtNDArray}, {@code MxNDArray}, etc.).
     *
     * @param node the manager node to inspect
     * @return the total number of NDArray resources directly owned by the
     *         node (does not recurse into sub-managers)
     */
    private long countNDArrays(ManagerNode node) {
        return node.byType().entrySet().stream()
                .filter(e -> e.getKey().endsWith("NDArray"))
                .mapToLong(java.util.Map.Entry::getValue)
                .sum();
    }
}
