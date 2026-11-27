package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement.ManagerNode;
import br.com.guialves.rflr.execs.base.*;
import br.com.guialves.rflr.execs.games.atari.AgentCNNDQNMain;
import br.com.guialves.rflr.execs.games.atari.AgentCNNRainbowDQNMain;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.getDebugDump;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.systemResourceCount;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test for memory leaks across all agent training pipelines.
 *
 * <h2>How a leak is detected</h2>
 * <p>A <b>single</b> snapshot of the manager tree cannot prove that there is no leak: a leak is
 * a <b>growth</b> over the training, and a leak that is released at the end of the episode is
 * invisible in the last snapshot. The agent collects a snapshot of the tree at <b>every step</b>
 * (see {@code AbstractAgent.managerNodes}), so this test works on the <b>time series</b> of
 * snapshots and asserts that:</p>
 *
 * <ol>
 *   <li><b>the part of the tree that is not the replay buffer does not grow</b>. The buffer is
 *       expected to fill up; everything else (the networks, the models, the managers of the
 *       training step, the copies of the observations) must stay flat. This is the assertion that
 *       catches the leaks: a leaked copy, a leaked manager or a leaked batch makes this number
 *       grow step after step;</li>
 *   <li>the same for the number of sub-managers of the tree;</li>
 *   <li>the buffer is bounded by its capacity (2 NDArrays per experience);</li>
 *   <li>the structure of the tree is the expected one (2 networks with the expected number of
 *       parameters, and the expected number of managers owned by the buffer).</li>
 * </ol>
 *
 * <h2>Why the buffer is not counted as a leak</h2>
 * <p>The replay buffer is <b>expected</b> to grow: it stores the experiences until it reaches its
 * capacity. The managers of the buffer are therefore identified by name and excluded from the
 * growth check. This also solves the {@code NEnvReplayBuffer} case: it owns one buffer per
 * environment, and those buffers are <b>siblings</b> of the container in the manager tree, not
 * children of it, so they have to be collected by name too.</p>
 *
 * <h2>Slack</h2>
 * <p>The growth checks allow a small slack, because the temporaries of the training step are
 * alive at the moment of the snapshot (the mini batch, the activations, the copies of the
 * observations). The slack is measured, not guessed: see {@link #MAX_GROWTH}.</p>
 *
 * <h2>Warm up</h2>
 * <p>The first snapshots are taken while the environment is still warming up (the vectorized
 * environment creates the initial observation of the 64 environments after the first step). That
 * one time allocation is not part of the comparison: see {@link #WARM_UP_FRACTION}.</p>
 */
@DisplayName("Memory leak regression tests")
class MemoryLeakTest {

    private static final String FRAMES_LIMIT = "2500";
    private static final String BUFFER_CAPACITY = "128";
    private static final String DONT_SAVE_MODEL = "false";
    private static final String DONT_RECORD = "false";
    private static final String DONT_SHOW_METRICS = "false";
    private static final String DONT_RENDER = "false";
    private static final String DONT_SAVE_CONFIG = "false";

    private static final int GC_TRIES = 3;
    private static final int MIN_SLEEP = 100;

    /**
     * The maximum growth, in NDArrays or in managers, of the part of the tree that does not
     * belong to the replay buffer.
     *
     * <p>Measured on the fixed code: the temporaries of the training step are alive at the
     * moment of the snapshot, so the number moves a little between two snapshots. A real leak
     * grows by at least one resource per step, which is orders of magnitude bigger.</p>
     */
    private static final int MAX_GROWTH = 16;

    /**
     * The fraction of the measured series used as the "beginning" and as the "end" of the
     * training when the trend of the series is measured.
     *
     * <p>The trend is the difference between the median of the first third and the median of
     * the last third of the series. The median is used instead of the minimum and the maximum
     * because a single snapshot taken at a different point of the lifecycle (the one taken
     * after the training, for example) is an outlier, and the minimum and the maximum would
     * report it as a growth of the whole series.</p>
     */
    private static final double TREND_WINDOW = 1.0 / 3.0;

    /**
     * The fraction of the snapshots at the beginning of the training that is ignored by the
     * growth checks.
     *
     * <p>The first snapshots are taken while the environment is still warming up: the vectorized
     * environment, for example, creates the initial observation of the 64 environments after the
     * first step, so the tree jumps by 64 NDArrays between the first and the second snapshot and
     * stays flat afterwards. That jump is a one time allocation, not a leak, and it would be
     * reported as one if the first snapshots were part of the comparison.</p>
     *
     * <p>A real leak keeps growing during the whole training, so it is still detected: the
     * resources that are leaked accumulate step after step, and the growth of the second half of
     * the training is much bigger than the allowed slack.</p>
     */
    private static final double WARM_UP_FRACTION = 0.10;

    /**
     * The minimum number of snapshots needed to compare the beginning of the training with its
     * end. A single snapshot proves nothing.
     */
    private static final int MIN_SNAPSHOTS = 3;

    /** DeepQNetworkMLP has 3 Linear layers (weight + bias each). */
    private static final int DEEP_NET_PARAMS = 6;

    /** DeepQNetworkCNN has 3 Linear layers (weight + bias each). */
    private static final int DEEP_CNN_NET_PARAMS = 10;

    /** DuelingQNetworkMLP has 4 Linear layers (2 backbone + 1 value + 1 advantage). */
    private static final int DUELING_NET_PARAMS = 8;

    /** NoisyQNetworkMLP has 1 Linear + 2 NoisyLayer (each NoisyLayer = 4 params). */
    private static final int NOISY_NET_PARAMS = 10;

    /** NoisyDuelingQNetworkMLP has 1 Linear + 4 NoisyLayer. */
    private static final int NOISY_DUELING_NET_PARAMS = 18;

    /** CategoricalQNetworkMLP has 3 Linear layers (weight + bias each). */
    private static final int C51_DQN_NET_PARAMS = 7;

    /** RainbowQNetworkMLP has 3 Linear layers (weight + bias each). */
    private static final int RAINBOW_DQN_NET_PARAMS = 19;

    /** RainbowQNetworkCNN has 4 CNN Layers + 3 Linear layers (weight + bias each). */
    private static final int RAINBOW_DQN_CNN_NET_PARAMS = 23;

    /**
     * Each stored experience contributes exactly 2 NDArrays to its buffer: one {@code state}
     * and one {@code nextState}.
     */
    private static final int NDARRAYS_PER_EXPERIENCE = 2;

    /**
     * The expectation of one agent: the names of the managers owned by the replay buffer, the
     * prefix of the networks, how many parameters each network has and how many managers the
     * buffer owns.
     *
     * @param agentName             the name of the agent, used in the messages
     * @param networkPrefix         the prefix of the name of the managers of the networks
     * @param expectedNetworkParams the number of NDArrays (parameters) of each network
     * @param bufferPrefixes        the prefixes of the names of the managers owned by the buffer
     * @param expectedBufferManagers the number of managers owned by the buffer
     * @param nStep                 the n-step of the buffer, used to bound the buffer size
     * @param numberOfEnvs          the number of environments of the agent
     */
    private record AgentSpec(String agentName,
                             String networkPrefix,
                             int expectedNetworkParams,
                             List<String> bufferPrefixes,
                             int expectedBufferManagers,
                             int nStep,
                             int numberOfEnvs) {

        boolean isBufferManager(String name) {
            return bufferPrefixes.stream().anyMatch(name::startsWith);
        }
    }

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
        System.setProperty("agent.saveConfig", DONT_SAVE_CONFIG);
        // The assertions below read the NDManager tree snapshots, so the dump must be
        // enabled explicitly instead of relying on the default of each runner.
        System.setProperty("agent.debugMemoryLeak", "true");
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
    @DisplayName("AgentDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentDQN() {
        assertNoLeak(AgentDQNMain.run(), singleBuffer("AgentDQN", "DeepQNetworkMLP-",
                DEEP_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentCNNDQNMain should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentCNNDQN() {
        assertNoLeak(AgentCNNDQNMain.run(), singleBuffer("AgentCNNDQN", "DeepQNetworkCNN-",
                DEEP_CNN_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentDDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentDDQN() {
        assertNoLeak(AgentDDQNMain.run(), singleBuffer("AgentDDQN", "DeepQNetworkMLP-",
                DEEP_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentNStepDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentNStepDQN() {
        assertNoLeak(AgentNStepDQNMain.run(), singleBuffer("AgentNStepDQN", "DeepQNetworkMLP-",
                DEEP_NET_PARAMS, "NStepExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentDuelingDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentDuelingDQN() {
        assertNoLeak(AgentDuelingDQNMain.run(), singleBuffer("AgentDuelingDQN", "DuelingQNetworkMLP-",
                DUELING_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentDQNPER should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentDQNPER() {
        assertNoLeak(AgentDQNPERMain.run(), singleBuffer("AgentDQNPER", "DeepQNetworkMLP-",
                DEEP_NET_PARAMS, "PrioritizedReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentNoisyNetDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentNoisyNetDQN() {
        assertNoLeak(AgentNoisyNetDQNMain.run(), singleBuffer("AgentNoisyNetDQN", "NoisyQNetworkMLP-",
                NOISY_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentNoisyDuelingNetDDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentNoisyDuelingNetDDQN() {
        assertNoLeak(AgentNoisyDuelingNetDDQNMain.run(), singleBuffer("AgentNoisyDuelingNetDDQN",
                "NoisyDuelingQNetworkMLP-", NOISY_DUELING_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentC51DQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentC51DQN() {
        assertNoLeak(AgentC51DQNMain.run(), singleBuffer("AgentC51DQN", "CategoricalQNetworkMLP-",
                C51_DQN_NET_PARAMS, "ExperienceReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentRainbowDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentRainbowDQN() {
        assertNoLeak(AgentRainbowDQNMain.run(), singleBuffer("AgentRainbowDQN", "RainbowQNetworkMLP-",
                RAINBOW_DQN_NET_PARAMS, "NStepPrioritizedReplayBuffer-", 1));
    }

    @Test
    @DisplayName("AgentCNNRainbowDQN should not leak memory during the training")
    void shouldNotLeakMemoryAfterAgentCNNRainbowDQN() {
        System.setProperty("agent.framesLimit", "100000");
        System.setProperty("agent.bufferCapacity", "1024");

        // The NEnvReplayBuffer owns one buffer per environment, and the buffers of the
        // environments are SIBLINGS of the container in the manager tree (they are created
        // with the parent of the container). They are therefore collected by name too, and
        // their number is derived from the configuration instead of being a magic number.
        int numberOfEnvs = Integer.getInteger("agent.numberOfEnvs", 64);
        var spec = new AgentSpec("AgentCNNRainbowDQN", "RainbowQNetworkCNN-",
                RAINBOW_DQN_CNN_NET_PARAMS,
                List.of("NEnvReplayBuffer-", "NStepPrioritizedReplayBuffer-"),
                1 + numberOfEnvs,
                3,
                numberOfEnvs);
        assertNoLeak(AgentCNNRainbowDQNMain.run(), spec);
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
     * The expectation of an agent with a single buffer.
     */
    private AgentSpec singleBuffer(String agentName,
                                   String networkPrefix,
                                   int expectedNetworkParams,
                                   String bufferPrefix,
                                   int expectedBufferManagers) {
        return new AgentSpec(agentName, networkPrefix, expectedNetworkParams,
                List.of(bufferPrefix), expectedBufferManagers, 1, 1);
    }

    /**
     * The classification of one snapshot of the manager tree.
     *
     * @param bufferNDArrays the NDArrays owned by the managers of the replay buffer
     * @param otherNDArrays  the NDArrays owned by everything else (the networks, the models,
     *                        the temporaries of the training step, ...)
     * @param bufferManagers the number of managers owned by the replay buffer
     * @param otherManagers  the number of managers owned by everything else
     */
    private record TreeStats(long bufferNDArrays,
                             long otherNDArrays,
                             long bufferManagers,
                             long otherManagers) {
    }

    /**
     * Asserts that the training of an agent does not leak memory.
     *
     * @param snapshots the snapshots of the manager tree taken during the training
     * @param spec      the expectation of the agent
     */
    private void assertNoLeak(List<ManagerNode> snapshots, AgentSpec spec) {
        assertTrue(snapshots.size() >= MIN_SNAPSHOTS,
                spec.agentName() + ": the leak check compares the beginning of the training with "
                        + "its end, so it needs at least " + MIN_SNAPSHOTS + " snapshots, got "
                        + snapshots.size() + ". Use more frames (agent.framesLimit) so that the "
                        + "training is spread over several snapshots.");

        var stats = snapshots.stream().map(snapshot -> classify(snapshot, spec)).toList();

        assertStructure(snapshots.getLast(), spec);

        // The warm up of the environment is not part of the comparison: it is a one time
        // allocation, not a growth.
        //
        // The last snapshot is not part of it either: it is taken when the training is over,
        // OUTSIDE of a step, when the temporaries of the step are already closed. In the
        // vectorized training a step holds the observations of the 64 environments (177
        // NDArrays), so that snapshot has less resources BY CONSTRUCTION and it is not
        // comparable with the snapshots taken during the training.
        int warmUp = Math.min(stats.size() - 1, (int) Math.ceil(stats.size() * WARM_UP_FRACTION));
        int end = Math.max(warmUp + 1, stats.size() - 1);
        var measured = stats.subList(warmUp, end);

        // LEAK 1: everything that is not the buffer must stay flat.
        //
        // A leak shows up in the series in one of two ways, and the check uses one robust
        // statistic for each of them, instead of min/max (which a single outlier blows up:
        // the post training snapshot of the vectorized training has 48 NDArrays against the
        // 177 of the training, and min/max reported that difference as a leak):
        //
        //   - SPREAD: the leak grows inside the episode and is released with it (a saw
        //     tooth). The spread of the middle half of the series (the interquartile
        //     range) sees the amplitude of the saw tooth and ignores the outliers;
        //   - TREND: the leak grows during the whole training. The difference between the
        //     medians of the first and of the last third of the series sees it.
        var otherArrays = measured.stream().map(TreeStats::otherNDArrays).sorted().toList();
        long spreadArrays = spread(otherArrays);
        long trendArrays = trend(otherArrays);
        assertTrue(spreadArrays <= MAX_GROWTH && trendArrays <= MAX_GROWTH,
                spec.agentName() + ": the NDArrays that do not belong to the replay buffer are not "
                        + "stable: the spread of the series is " + spreadArrays
                        + " and the growth between the beginning and the end of the training is "
                        + trendArrays + " (allowed growth: " + MAX_GROWTH + "). This is a memory leak: "
                        + "something created during the training is not being closed. Snapshots: " + stats);

        // LEAK 2: the same for the managers: a leaked manager per step is a leak too.
        var otherManagers = measured.stream().map(TreeStats::otherManagers).sorted().toList();
        long spreadManagers = spread(otherManagers);
        long trendManagers = trend(otherManagers);
        assertTrue(spreadManagers <= MAX_GROWTH && trendManagers <= MAX_GROWTH,
                spec.agentName() + ": the managers that do not belong to the replay buffer are not "
                        + "stable: the spread of the series is " + spreadManagers
                        + " and the growth between the beginning and the end of the training is "
                        + trendManagers + " (allowed growth: " + MAX_GROWTH + "). This is a memory leak: "
                        + "a sub manager is created and not closed. Snapshots: " + stats);

        // BOUND: the buffer is bounded by its capacity.
        int capacity = Integer.getInteger("agent.bufferCapacity", Integer.parseInt(BUFFER_CAPACITY));
        long maxBufferArrays = stats.stream().mapToLong(TreeStats::bufferNDArrays).max().orElseThrow();
        long maxExpected = (long) capacity * NDARRAYS_PER_EXPERIENCE
                + (long) spec.numberOfEnvs() * spec.nStep() * NDARRAYS_PER_EXPERIENCE
                + MAX_GROWTH;
        assertTrue(maxBufferArrays <= maxExpected,
                spec.agentName() + ": the replay buffer holds " + maxBufferArrays
                        + " NDArrays, which is more than the " + maxExpected
                        + " allowed by its capacity (" + capacity + " experiences, "
                        + NDARRAYS_PER_EXPERIENCE + " NDArrays each, plus the n-step window). "
                        + "Snapshots: " + stats);

        // STRUCTURE: the managers of the buffer are the expected ones.
        long bufferManagers = stats.getLast().bufferManagers();
        assertEquals(spec.expectedBufferManagers(), bufferManagers,
                spec.agentName() + ": expected exactly " + spec.expectedBufferManagers()
                        + " managers owned by the replay buffer (prefixes " + spec.bufferPrefixes()
                        + "), got " + bufferManagers + " with byType=" + stats.getLast());
    }

    /**
     * The spread of the middle half of a series (the interquartile range).
     *
     * <p>It is the amplitude of the saw tooth of a leak that grows inside the episode and is
     * released with it, and it ignores the outliers: a single value outside the series
     * does not change the quartiles.</p>
     *
     * @param sorted the series, sorted in ascending order
     * @return the interquartile range of the series
     */
    private static long spread(List<Long> sorted) {
        return percentile(sorted, 0.75) - percentile(sorted, 0.25);
    }

    /**
     * The growth of a series between its beginning and its end.
     *
     * <p>It is the difference between the median of the first third and the median of the last
     * third of the series: a leak that grows during the whole training moves the medians, and
     * an outlier does not.</p>
     *
     * @param sorted the series, sorted in ascending order
     * @return the growth of the series
     */
    private static long trend(List<Long> sorted) {
        int window = Math.max(1, (int) Math.ceil(sorted.size() * TREND_WINDOW));
        long head = median(sorted.subList(0, window));
        long tail = median(sorted.subList(sorted.size() - window, sorted.size()));
        return tail - head;
    }

    /**
     * The median of a sorted series.
     *
     * @param sorted the series, sorted in ascending order
     * @return the median of the series
     */
    private static long median(List<Long> sorted) {
        if (sorted.isEmpty()) throw new IllegalArgumentException("the series is empty");
        int middle = sorted.size() / 2;
        return (sorted.size() % 2 == 0)
                ? (sorted.get(middle - 1) + sorted.get(middle)) / 2
                : sorted.get(middle);
    }

    /**
     * A percentile of a sorted series (nearest rank: the value at the given position).
     *
     * @param sorted   the series, sorted in ascending order
     * @param fraction the percentile, between 0 and 1
     * @return the value of the percentile
     */
    private static long percentile(List<Long> sorted, double fraction) {
        if (sorted.isEmpty()) throw new IllegalArgumentException("the series is empty");
        int index = (int) Math.round((sorted.size() - 1) * fraction);
        return sorted.get(Math.clamp(index, 0, sorted.size() - 1));
    }

    /**
     * Asserts the structure of the manager tree at the end of the training.
     *
     * @param root the snapshot of the root of the manager tree
     * @param spec the expectation of the agent
     */
    private void assertStructure(ManagerNode root, AgentSpec spec) {
        var networks = root.findMatching(n -> n.name().startsWith(spec.networkPrefix()) ||
                n.name().startsWith("clone-" + spec.networkPrefix()));
        assertEquals(2, networks.size(),
                spec.agentName() + ": expected exactly 2 sub-managers starting with '"
                        + spec.networkPrefix() + "', found " + networks.size());
        for (ManagerNode net : networks) {
            long paramCount = countNDArrays(net);
            assertEquals(spec.expectedNetworkParams(), paramCount,
                    spec.agentName() + ": network '" + net.name()
                            + "' should have exactly " + spec.expectedNetworkParams()
                            + " parameters, got " + paramCount
                            + " NDArrays (byType=" + net.byType() + ")");
        }
    }

    /**
     * Classifies a snapshot of the manager tree: the resources owned by the replay buffer and
     * the resources owned by everything else.
     *
     * <p>Each NDArray is attributed to the closest manager that owns it, so a buffer nested
     * in another buffer is not counted twice.</p>
     *
     * @param node the root of the snapshot
     * @param spec the expectation of the agent
     * @return the classification of the snapshot
     */
    private TreeStats classify(ManagerNode node, AgentSpec spec) {
        long bufferNDArrays = 0;
        long otherNDArrays = 0;
        long bufferManagers = 0;
        long otherManagers = 0;
        for (ManagerNode current : flatten(node)) {
            if (spec.isBufferManager(current.name())) {
                ++bufferManagers;
                bufferNDArrays += countNDArrays(current);
            } else {
                ++otherManagers;
                otherNDArrays += countNDArrays(current);
            }
        }
        return new TreeStats(bufferNDArrays, otherNDArrays, bufferManagers, otherManagers);
    }

    /**
     * Flattens the manager tree, in pre-order.
     */
    private List<ManagerNode> flatten(ManagerNode node) {
        var all = new ArrayList<ManagerNode>();
        collect(node, all);
        return all;
    }

    private void collect(ManagerNode node, List<ManagerNode> all) {
        all.add(node);
        for (var child : node.children()) {
            collect(child, all);
        }
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
                .mapToLong(Map.Entry::getValue)
                .sum();
    }
}
