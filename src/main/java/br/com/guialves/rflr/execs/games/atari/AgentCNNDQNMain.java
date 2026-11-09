package br.com.guialves.rflr.execs.games.atari;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkCNN;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.execs.RLConfig;
import br.com.guialves.rflr.execs.RLRunOptions;
import br.com.guialves.rflr.execs.RLRunner;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.PreProcessingWrapper;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.List;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Atari <b>Pong</b> ({@code ALE/Pong-v5}) trained with a convolutional DQN.
 *
 * <h2>Hyperparameters</h2>
 * The values below follow the modern Atari DQN protocol. The primary source is the
 * CleanRL reference implementation, which documents every default against the
 * original Nature DQN table:
 * <ul>
 *   <li><a href="https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py">cleanrl/dqn_atari.py</a></li>
 *   <li><a href="https://docs.cleanrl.dev/rl-algorithms/dqn/">CleanRL - Deep Q-Learning (DQN)</a></li>
 * </ul>
 * <table>
 *   <caption>Hyperparameters and their sources</caption>
 *   <tr><th>Hyperparameter</th><th>Value</th><th>Source</th></tr>
 *   <tr><td>Environment</td><td>{@code ALE/Pong-v5}</td>
 *       <td>ALE v5: {@code frameskip=4}, {@code repeat_action_probability=0.25}</td></tr>
 *   <tr><td>Actions</td><td>6 (minimal action set)</td>
 *       <td>Mnih et al. 2015 - Nature DQN</td></tr>
 *   <tr><td>Observation</td><td>84x84 grayscale, 4 frames stacked</td>
 *       <td>Hessel et al. 2018 - Rainbow, Table 3</td></tr>
 *   <tr><td>Optimizer</td><td>Adam, lr 1e-4</td>
 *       <td>CleanRL dqn_atari.py (Nature used RMSProp 2.5e-4)</td></tr>
 *   <tr><td>Discount factor</td><td>0.99</td>
 *       <td>Rainbow Table 4 / Nature DQN</td></tr>
 *   <tr><td>Minibatch size</td><td>32</td>
 *       <td>Rainbow Table 4</td></tr>
 *   <tr><td>Replay buffer</td><td>100,000 transitions (paper: 1,000,000)</td>
 *       <td>Rainbow Table 4 - reduced to fit the GPU memory</td></tr>
 *   <tr><td>Exploration</td><td>epsilon 1.0 -&gt; 0.01 over the first 10% of the steps</td>
 *       <td>CleanRL dqn_atari.py (exploration_fraction = 0.10)</td></tr>
 *   <tr><td>Target network</td><td>copied every 1,000 agent steps</td>
 *       <td>CleanRL dqn_atari.py (target_network_frequency = 1000)</td></tr>
 *   <tr><td>Training budget</td><td>200,000,000 emulator frames</td>
 *       <td>Nature DQN and Rainbow (CleanRL uses 40M as a cheaper option)</td></tr>
 *   <tr><td>Q network</td><td>32/64/64 conv + 512 dense</td>
 *       <td>Rainbow Table 4 - already used by {@link DeepQNetworkCNN}</td></tr>
 * </table>
 *
 * <h2>How many frames does one agent step consume?</h2>
 * The budget of the papers is counted in <b>emulator frames</b>, and the ALE v5
 * environment repeats the action <b>4 times</b> inside the env
 * ({@code frameskip: 4} in the registration). On top of that, the
 * {@link PreProcessingWrapper} takes {@code FRAMES_SKIP} env steps for each of the
 * {@code CHANNELS} stacked frames. Therefore:
 * <pre>
 * emulator frames per agent step = FRAMES_SKIP * CHANNELS = 4 * 4 = 16
 * </pre>
 * Measured with {@code ale.getFrameNumber()}: one {@code env.step()} of
 * {@code ALE/Pong-v5} advances the emulator by 4 frames.
 * <p>
 * The training loop counts {@code framesSkip} per agent step, so the counter used by
 * {@code framesLimit} is <b>16x smaller</b> than the emulator frames:
 * <pre>
 * framesLimit = (TRAINING_FRAMES / 64) * FRAMES_SKIP = 12,500,000 for 200M frames
 * </pre>
 *
 * <h2>Known deviations from the paper</h2>
 * <ul>
 *   <li>The action is repeated 64 times per decision instead of 4, so the agent
 *   decides 16x less often than in the paper and performs 16x fewer gradient updates
 *   per emulator frame (the paper updates every 4 frames, this code updates every 64).</li>
 *   <li>The replay buffer lives in the GPU (the parent manager follows the training
 *   device), so 1M transitions would need ~56 GB of VRAM. The default is 100,000
 *   (~5.6 GB).</li>
 *   <li>There is no no-op start, no terminal on life loss and no reward clipping.
 *   For Pong the reward clipping is a no-op (the reward is already -1/0/+1) and the
 *   modern protocol disables the life loss heuristic.</li>
 * </ul>
 *
 * <h2>Other Pong environments of ALE</h2>
 * All of them are registered by ale_py and were measured in this environment
 * (ale_py 0.12.0 / gymnasium 1.2.2):
 * <table>
 *   <caption>ALE registrations and their frame semantics</caption>
 *   <tr><th>Id</th><th>frameskip</th><th>sticky actions</th><th>frames per env.step()</th></tr>
 *   <tr><td>{@code ALE/Pong-v5}</td><td>4</td><td>0.25</td><td>4</td></tr>
 *   <tr><td>{@code PongNoFrameskip-v4}</td><td>1</td><td>0.0</td><td>1</td></tr>
 *   <tr><td>{@code Pong-v4}</td><td>(2, 5) random</td><td>0.0</td><td>4</td></tr>
 *   <tr><td>{@code PongNoFrameskip-v0}</td><td>1</td><td>0.25</td><td>1</td></tr>
 * </table>
 * {@code PongNoFrameskip-v4} is the classic DQN setup (the frame skip is done by the
 * preprocessing, not by the env, and there are no sticky actions). Using it with the
 * gymnasium wrappers listed below makes one agent step consume exactly 4 emulator
 * frames, like in the paper.
 *
 * <h2>References</h2>
 * <ul>
 *   <li>Mnih et al. 2015 - Human-level control through deep reinforcement learning.
 *       <a href="https://www.nature.com/articles/nature14236">Nature 518</a></li>
 *   <li>Hessel et al. 2018 - Rainbow: Combining Improvements in Deep Reinforcement
 *       Learning. <a href="https://arxiv.org/abs/1710.02298">arXiv:1710.02298</a></li>
 *   <li>Hessel et al. 2018 - Benchmarking Deep Reinforcement Learning for Atari Games.
 *       <a href="https://arxiv.org/abs/1707.06887">arXiv:1707.06887</a></li>
 *   <li>Pohlen et al. 2018 - Observe and Look Further: Achieving Consistent
 *       Performance on Atari. <a href="https://arxiv.org/abs/1805.11593">arXiv:1805.11593</a></li>
 *   <li>Machado et al. 2018 - Revisiting the Arcade Learning Environment.
 *       <a href="https://arxiv.org/abs/1709.06009">arXiv:1709.06009</a></li>
 *   <li>Google Dopamine baselines - standardized Atari configuration.
 *       <a href="https://github.com/google/dopamine/blob/master/baselines/atari/README.md">baselines/atari</a></li>
 * </ul>
 */
public class AgentCNNDQNMain {

    private static final String ENV_NAME = "PongNoFrameskip-v4";
    private static final String ALGORITHM_NAME = "pong_cnn_dqn";
    private static final int OBSERVATIONS = 84;
    private static final int CHANNELS = 4;
    /** Minimal action set of Pong: NOOP, FIRE, LEFT, RIGHT, LEFTFIRE, RIGHTFIRE. */
    private static final int ACTIONS = 6;
    private static final float LEARNING_RATE = 1e-4f;
    private static final float DISCOUNT_FACTOR = 0.99f;
    private static final int BATCH_SIZE = 32;
    private static final int FRAMES_SKIP = 4;
    private static final float MAX_EPSILON = 1.0f;
    private static final float MIN_EPSILON = 0.01f;
    private static final float EXPLORATION_FRACTION = 0.10f;
    private static final int TARGET_NETWORK_FREQUENCY_STEPS = 1_000;

    /**
     * Training budget in <b>emulator frames</b>: 200M is the budget of the original
     * Nature DQN and of Rainbow. CleanRL uses 40M frames (10M steps) as a cheaper
     * alternative.
     */
    private static final int TRAINING_FRAMES = 200_000_000;

    /**
     * Replay buffer capacity. The paper uses 1M transitions, but the buffer is kept in
     * the GPU memory (~56 KB per transition), which does not fit in a 24 GB GPU.
     */
    private static final int BUFFER_CAPACITY = 100_000;

    public static void main() {
        run();
    }

    static void main(String[] args) {
        var opts = parseArgs(args, AgentCNNDQNMain.class.getSimpleName());
        run(opts);
    }

    public static List<DJLMemoryManagement.ManagerNode> run() {
        return run(RLRunOptions.defaults());
    }

    public static List<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {

        var framesPerStep = FRAMES_SKIP * CHANNELS;
        var defaultSteps = TRAINING_FRAMES / framesPerStep;

        var framesLimit = getIntProp("agent.framesLimit",
                String.valueOf(defaultSteps * FRAMES_SKIP));
        var totalSteps = Math.max(1, framesLimit / FRAMES_SKIP);
        var annealSteps = Math.max(1, (int) (totalSteps * EXPLORATION_FRACTION));

        var builder = RLConfig.builder()
                .envName(ENV_NAME)
                .runnerClass(AgentCNNDQNMain.class.getSimpleName())
                .algorithmName(ALGORITHM_NAME)
                .observations(OBSERVATIONS) // 84x84
                .channels(CHANNELS)
                .actions(ACTIONS)
                .learningRate(LEARNING_RATE)
                .maxEpsilon(MAX_EPSILON)
                .minEpsilon(MIN_EPSILON)
                .epsilonDecay((MAX_EPSILON - MIN_EPSILON) / annealSteps)
                .discountFactor(DISCOUNT_FACTOR) // gamma
                .updateQTargetAtTimeN(TARGET_NETWORK_FREQUENCY_STEPS * FRAMES_SKIP)
                .batchSize(BATCH_SIZE)
                .framesSkip(FRAMES_SKIP)
                .resize(OBSERVATIONS)
                .concatenate(CHANNELS)
                .importLibs(List.of("ale_py; gym.register_envs(ale_py)"))
                .framesLimit(framesLimit)
                .bufferCapacity(getIntProp("agent.bufferCapacity",
                        String.valueOf(BUFFER_CAPACITY)))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "false"))
                .renderRun(getBoolProp("agent.renderRun", "true"))
                .runMaxTries(getIntProp("agent.maxTries", "1"));

        if (opts.loadModelPrefix() != null) {
            builder = builder.loadModelPrefix(opts.loadModelPrefix());
        }

        var config = builder.build();

        return RLRunner.run(config, (env, optimizer, plotTrackers, parent) -> {
            var preProcessing = new PreProcessingWrapper(env, config.framesSkip(), config.resize(), config.concatenate());
            return buildDQN(config, preProcessing, optimizer, plotTrackers, parent);
        });
    }

    /*
     * Alternative pipeline: the preprocessing runs in Python (gymnasium wrappers),
     * which is the one described in the paper - the action is repeated 4 times and the
     * stack holds the last 4 decisions. With an env that has frameskip=1
     * (PongNoFrameskip-v4) it makes the frame accounting exact: one agent step
     * consumes 4 emulator frames, and framesLimit can be read as the number of
     * emulator frames of the paper. It moves the grayscale/resize to the Python side.
     *
     *     .wrappers(List.of(
     *             new MaxAndSkipObservation(4),
     *             new GrayscaleObservation(),
     *             new ResizeObservation(new int[] {84, 84}),
     *             new FrameStackObservation(4)
     *     ))
     *
     * and buildDQN(config, env, optimizer, plotTrackers, parent) with the raw env,
     * instead of the PreProcessingWrapper used above.
     */

    private static IAgent buildDQN(RLConfig config,
                                   IEnv env,
                                   Optimizer optimizer,
                                   PlotTrackers plotTrackers,
                                   NDManager parent) {
        boolean loadModel = config.loadModelPrefix() != null;
        Supplier<IDeepQNetwork> networkFactory = loadModel
                ? () -> new DeepQNetworkCNN(
                        config.channels(),
                        config.observations(),
                        config.actions(),
                        config.path(),
                        config.loadModelPrefix(),
                        parent)
                : () -> new DeepQNetworkCNN(
                        config.channels(),
                        config.observations(),
                        config.actions(),
                        parent);

        return new AgentDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                parent,
                networkFactory,
                plotTrackers,
                config.debugMemoryLeak()
        );
    }
}
