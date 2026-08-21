package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.NStepPrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.RainbowQNetworkMLP;
import br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.algorithms.rainbowdqn.AgentRainbowDQN;
import br.com.guialves.rflr.algorithms.rainbowdqn.CategoricalCrossEntropyPERLoss;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.Optional;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;
import static java.lang.System.getProperty;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentRainbowDQNMain {

    static void main() {
        run();
    }

    static void main(String[] args) {
        var opts = parseArgs(args, AgentRainbowDQNMain.class.getSimpleName());
        run(opts);
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {
        return run(RLRunOptions.defaults());
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {
        var builder = RLConfig.builder()
                .envName("LunarLander-v3")
                .runnerClass(AgentRainbowDQNMain.class.getSimpleName())
                .algorithmName("rainbow_dqn")
                .observations(8)
                .actions(4)
                .alpha(0.2f)
                .beta(0.6f)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(1000)
                .batchSize(128)
                .nStep(3)
                .atoms(50)
                .vMin(-10.0f)
                .vMax(+10.0f)
                .duelingType(DuelingType.valueOf(getProperty("agent.duelingType", "MEAN")))
                .framesLimit(getIntProp("agent.framesLimit", "300000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "30000"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "true"))
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
                return new CategoricalCrossEntropyPERLoss();
            }

            @Override
            public IReplayBuffer replayBuffer(RLConfig config, NDManager manager) {
                return new NStepPrioritizedReplayBuffer(
                        config.nStep(),
                        config.discountFactor(),
                        config.alpha(),
                        config.bufferCapacity(),
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
                ? () -> new RainbowQNetworkMLP(
                        config.observations(),
                        config.actions(),
                        new CategoricalBellmanProjection(
                                config.atoms(), config.vMin(), config.vMax()),
                        config.path(),
                        config.loadModelPrefix(),
                        parent,
                        config.duelingType())
                : () -> new RainbowQNetworkMLP(
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
