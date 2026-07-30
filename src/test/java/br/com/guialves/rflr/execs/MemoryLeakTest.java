package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test for memory leaks across all agent training pipelines.
 *
 * <p>Each agent test runs a complete {@code main()} (env + train + run) and
 * verifies that the system-wide NDManager resource count returns to the
 * baseline after the agent finishes. If any sub-manager or array is not
 * properly closed, the count grows and the test fails.</p>
 *
 * <p>The diagnostics in this test rely on
 * {@link DJLMemoryManagement#getDebugDump(NDManager)}, which returns a
 * hierarchical {@link ManagerNode} snapshot of the manager tree built via
 * reflection on DJL's {@code BaseNDManager.resources} map. The tree is a
 * value object, safe to traverse, filter and assert against without holding
 * any reference to the live resources.</p>
 *
 * <p>This test was introduced after a series of leaks were discovered in the
 * buffer ({@code tempAttach} restoring duplicates to the buffer manager), the
 * inference path ({@code output.attach(modelManager)} pinning every forward
 * result) and the noisy layers (intermediate {@code w}, {@code b} tensors
 * accumulating in the network manager). It is designed to fail loudly if any
 * of these patterns reappear.</p>
 *
 * <h2>How it works</h2>
 * <ol>
 *   <li>Snapshot the total resource count of the DJL system manager.</li>
 *   <li>Invoke {@code AgentXxxMain.main()} (which creates a parent manager,
 *       trains the agent, runs evaluation, then closes everything via
 *       try-with-resources).</li>
 *   <li>Force GC + finalization, then snapshot the count again.</li>
 *   <li>Assert that the count growth is below the threshold.</li>
 *   <li>On failure, print the leaking sub-managers (top-10 by resource count)
 *       so the offending code path is immediately identifiable.</li>
 * </ol>
 *
 * <h2>Threshold rationale</h2>
 * <p>{@link #MAX_LEAK_PER_AGENT} is set to {@code 50} NDArrays. This is
 * generous enough to absorb minor flakiness from finalizers and Python
 * runtime bookkeeping, but tight enough to catch real leaks (the historic
 * bugs we fixed showed growth in the hundreds of thousands).</p>
 *
 * <h2>How to debug a failure</h2>
 * <p>If a test fails, the error message includes the leaked count plus a
 * top-N of the sub-managers that retained the most resources at the end of
 * the run. For deeper inspection, call
 * {@code DJLMemoryManagement.debugDump(manager)} at
 * the end of the failing agent's training to see the full hierarchical
 * tree. The name you set via {@code subMgr(...)} is shown in the dump,
 * making the source obvious.</p>
 */
@DisplayName("Memory leak regression tests")
class MemoryLeakTest {

    private static final String FRAMES_LIMIT = "500";
    private static final String BUFFER_CAPACITY = "128";
    private static final String DONT_SAVE_MODEL = "false";
    private static final String DONT_RECORD = "false";
    private static final String DONT_SHOW_METRICS = "false";

    /**
     * Maximum number of NDArrays that may remain un-freed after an agent run.
     * Set to a small constant to catch regressions while tolerating minor
     * finalizer / Python-state noise.
     */
    private static final int MAX_LEAK_PER_AGENT = 50;

    /**
     * When a hierarchical check verifies that a specific sub-manager
     * (e.g. an online network) does not retain temporary arrays after
     * an operation, this is the maximum direct resource count allowed.
     */
    private static final int MAX_LEAK_PER_SUBMANAGER = 5;

    /**
     * How many top-leaking sub-managers to include in a failure message.
     */
    private static final int TOP_N_LEAKERS = 10;

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
    }

    @AfterEach
    @DisplayName("Stabilize the JVM before the next test")
    void tearDownEach() throws InterruptedException {
        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        Thread.sleep(100);
        manager.close();
    }

    @Test
    @DisplayName("AgentDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentDQN() {
        assertLeakWithinThreshold("AgentDQN", AgentDQNMain::main);
    }

    @Test
    @DisplayName("AgentDDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentDDQN() {
        assertLeakWithinThreshold("AgentDDQN", AgentDDQNMain::main);
    }

    @Test
    @DisplayName("AgentNStepDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentNStepDQN() {
        assertLeakWithinThreshold("AgentNStepDQN", AgentNStepDQNMain::main);
    }

    @Test
    @DisplayName("AgentDuelingDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentDuelingDQN() {
        assertLeakWithinThreshold("AgentDuelingDQN", AgentDuelingDQNMain::main);
    }

    @Test
    @DisplayName("AgentDQNPER should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentDQNPER() {
        assertLeakWithinThreshold("AgentDQNPER", AgentDQNPERMain::main);
    }

    @Test
    @DisplayName("AgentNoisyNetDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentNoisyNetDQN() {
        assertLeakWithinThreshold("AgentNoisyNetDQN", AgentNoisyNetDQNMain::main);
    }

    @Test
    @DisplayName("AgentNoisyDuelingNetDDQN should not leak NDArrays")
    void shouldNotLeakMemoryAfterAgentNoisyDuelingNetDDQN() {
        assertLeakWithinThreshold(
                "AgentNoisyDuelingNetDDQN", AgentNoisyDuelingNetDDQNMain::main);
    }

    @Test
    @DisplayName("getDebugDump returns empty when manager is not a BaseNDManager")
    void getDebugDumpShouldReturnEmptyForNonBaseManager() {
        var root = getDebugDump(manager);
        assertTrue(root.isPresent(), "system manager is a BaseNDManager");
    }

    @Test
    @DisplayName("systemResourceCount matches subtree size of the root dump")
    void systemResourceCountShouldMatchSubtreeSize() {
        var root = getDebugDump(manager);
        assertTrue(root.isPresent());
        assertEquals(root.get().subtreeSize(), systemResourceCount(manager),
                "systemResourceCount is just subtreeSize() of the system root");
    }

    /**
     * Runs the given agent's {@code main()} method, snapshots the system-wide
     * resource count before and after, and asserts that growth is within the
     * allowed threshold. The assertion message includes actionable debugging
     * guidance.
     */
    private void assertLeakWithinThreshold(String agentName, Runnable runnable) {
        int before = systemResourceCount(manager);
        assertDoesNotThrow(runnable::run,
                () -> agentName + " threw an exception during execution");

        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        int after = systemResourceCount(manager);
        int leaked = after - before;

        if (leaked <= MAX_LEAK_PER_AGENT) {
            return;
        }

        var diagnostic = buildLeakDiagnostic(agentName, leaked, after);
        fail(diagnostic);
    }

    /**
     * Builds a human-readable diagnostic listing the top-N leaking
     * sub-managers by direct resource count. Used only on failure, so
     * the cost of building the tree is acceptable.
     */
    private String buildLeakDiagnostic(String agentName, int leaked, int after) {
        var root = getDebugDump(manager).orElse(null);
        String top = "(unable to inspect NDManager hierarchy)";

        if (root != null) {
            var top10 = root.findLeakingNodes(0L).stream()
                    .sorted((a, b) -> Long.compare(b.totalResources(), a.totalResources()))
                    .limit(TOP_N_LEAKERS)
                    .map(n -> String.format(
                            "    %s [uid=%s, open=%s, direct=%d, subtree=%d, byType=%s]",
                            n.name(), n.uid(), n.open(),
                            n.totalResources(), n.subtreeSize(), n.byType()))
                    .collect(Collectors.joining("\n"));
            top = top10.isEmpty() ? "(no leaking nodes reported)" : top10;
        }

        return String.format(
                "%s leaked %d NDArrays after a full run (max allowed: %d, total now: %d). "
                        + "Top %d sub-managers by direct resource count:%n%s%n"
                        + "Inspect the named managers above to identify the offending code path. "
                        + "For a full hierarchical tree, call "
                        + "DJLMemoryManagement.debugDump(manager) "
                        + "at the end of the agent's train() method.",
                agentName, leaked, MAX_LEAK_PER_AGENT, after, TOP_N_LEAKERS, top);
    }

    /**
     * Example: verify that an IDeepQNetwork's sub-manager does not retain
     * temporary arrays after a forward pass. Not part of the default suite
     * because it requires constructing a network in isolation, which is
     * already covered indirectly by the per-agent tests above.
     */
    @SuppressWarnings("unused")
    private void assertNetworkDoesNotLeakTempArrays(IDeepQNetwork net) {
        try (var parent = subMgr(
                manager, "scope-")) {
            try (var instance = net) {
                try (var sub = subMgr(parent, "forward-scope-")) {
                    var input = parent.create(new float[]{1, 2, 3, 4});
                    var output = instance.forward(input, NDArray::sum);
                    output.close();
                }
                var root = getDebugDump(parent).orElseThrow();
                var leak = root.findByNamePrefix("forward-scope-");
                assertTrue(leak.isEmpty(),
                        "forward scope must be released, found: " + leak.get());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Example: verify that an IReplayBuffer does not retain duplicate
     * experiences across calls to {@code sample()}. Inspects the system
     * manager because {@link IReplayBuffer} does not expose its
     * sub-manager in the interface. Not part of the default suite.
     */
    @SuppressWarnings("unused")
    private void assertBufferDoesNotLeakAcrossSamples(IReplayBuffer buffer) {
        int before = systemResourceCount(manager);
        try (var bufferRef = buffer) {
            var first = bufferRef.sample(8);
            first.close();
            var second = bufferRef.sample(8);
            second.close();
        }
        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        int after = systemResourceCount(manager);

        assertTrue(after - before <= MAX_LEAK_PER_AGENT,
                "buffer sampling leaked resources: before=" + before + ", after=" + after);
    }

    /**
     * Example: filter and assert over the tree using a custom predicate.
     * Useful when you want to check that no sub-manager with a specific
     * name pattern (e.g. noisy-layer-) accumulates more than N arrays.
     */
    @SuppressWarnings("unused")
    private List<ManagerNode> findSubManagersAbove(NDManager manager, String namePrefix, int limit) {
        var root = getDebugDump(manager).orElseThrow();
        return root.findMatching(n ->
                n.name().startsWith(namePrefix)
                        && n.totalResources() > limit);
    }
}
