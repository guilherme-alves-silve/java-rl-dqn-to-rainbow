package br.com.guialves.rflr.execs.games.atari;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.NStepPrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.c51dqn.CategoricalNLLPERLoss;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.RainbowQNetworkCNN;
import br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.algorithms.rainbowdqn.AgentRainbowDQN;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.execs.RLConfig;
import br.com.guialves.rflr.execs.RLRunOptions;
import br.com.guialves.rflr.execs.RLRunner;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.wrappers.FrameStackObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.GrayscaleObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.ResizeObservation;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.List;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;
import static java.lang.System.getProperty;

public class AgentCNNRainbowDQNMain {

    private static final String ENV_NAME = "ALE/Galaxian-v5";
    private static final String ALGORITHM_NAME = "galaxian_cnn_dqn";
    private static final int OBSERVATIONS = 84;
    private static final int CHANNELS = 4;
    private static final int ACTIONS = 6;
    private static final float LEARNING_RATE = 1e-4f;
    private static final float DISCOUNT_FACTOR = 0.99f;
    private static final int BATCH_SIZE = 256;
    private static final int FRAMES_SKIP = 4;
    private static final float MAX_EPSILON = 1.0f;
    private static final float MIN_EPSILON = 0.01f;
    private static final float EXPLORATION_FRACTION = 0.10f;
    private static final int TARGET_NETWORK_FREQUENCY_STEPS = 1_000;
    private static final float PRIORITIZED_REPLAY_ALPHA = 0.2f;
    private static final float PRIORITIZED_REPLAY_BETA = 0.6f;
    private static final int NUMBER_OF_VECTORIZED_ENVS = 64;

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
    private static final int BUFFER_CAPACITY = 1_000_000;

    static void main() {
        run();
    }

    static void main(String[] args) {
        var opts = parseArgs(args, AgentCNNRainbowDQNMain.class.getSimpleName());
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
                .runnerClass(AgentCNNRainbowDQNMain.class.getSimpleName())
                .algorithmName(ALGORITHM_NAME)
                .observations(OBSERVATIONS) // 84x84
                .wrappers(List.of(
                    // is disabled because ALE/Galaxian-v5 already skips 4 frames
                    // new MaxAndSkipObservation(4),
                    new GrayscaleObservation(),
                    new ResizeObservation(new int[] {OBSERVATIONS, OBSERVATIONS}),
                    new FrameStackObservation(4)
                ))
                .channels(CHANNELS)
                .actions(ACTIONS)
                .alpha(PRIORITIZED_REPLAY_ALPHA)
                .beta(PRIORITIZED_REPLAY_BETA)
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
                .vectorized(RLConfig.Vectorized.ASYNC)
                .numberOfEnvs(NUMBER_OF_VECTORIZED_ENVS)
                .importLibs(List.of("ale_py; gym.register_envs(ale_py)"))
                .framesLimit(framesLimit)
                .nStep(3)
                .atoms(51)
                .vMin(-200.0f)
                .vMax(+200.0f)
                .duelingType(DuelingType.valueOf(getProperty("agent.duelingType", "MEAN")))
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

        return RLRunner.run(config, new RLRunner.AgentFactory() {

            @Override
            public IAgent create(IEnv env, Optimizer optimizer, PlotTrackers plotTrackers, NDManager parent) {
                return buildRainbowDQN(config, env, optimizer, plotTrackers, parent);
            }

            @Override
            public Loss lossFunc() {
                return CategoricalNLLPERLoss.noneReduction();
            }

            @Override
            public IReplayBuffer replayBuffer(RLConfig config, NDManager manager) {
                return new NStepPrioritizedReplayBuffer(
                        config.bufferCapacity(),
                        config.nStep(),
                        config.discountFactor(),
                        config.alpha(),
                        manager
                );
            }
        });
    }

    private static IAgent buildRainbowDQN(RLConfig config,
                                          IEnv env,
                                          Optimizer optimizer,
                                          PlotTrackers plotTrackers,
                                          NDManager parent) {
        boolean loadModel = config.loadModelPrefix() != null;

        Supplier<IDeepQNetwork> networkFactory = loadModel
                ? () -> new RainbowQNetworkCNN(
                config.channels(),
                config.observations(),
                config.actions(),
                new CategoricalBellmanProjection(
                        config.atoms(), config.vMin(), config.vMax()),
                config.path(),
                config.loadModelPrefix(),
                parent,
                config.duelingType())
                : () -> new RainbowQNetworkCNN(
                config.channels(),
                config.observations(),
                config.actions(),
                config.atoms(),
                config.vMin(),
                config.vMax(),
                parent,
                config.duelingType());

        return new AgentRainbowDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                config.beta(),
                env,
                optimizer,
                parent,
                networkFactory,
                plotTrackers,
                config.debugMemoryLeak()
        );
    }
}
