package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.c51dqn.AgentC51NoisyNetDQN;
import br.com.guialves.rflr.algorithms.c51dqn.CategoricalNLLLoss;
import br.com.guialves.rflr.algorithms.networks.CategoricalQNoisyNetworkMLP;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.Optional;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentC51NoisyNetDQNMain {

    static void main() {
        run();
    }

    static void main(String[] args) {
        var opts = parseArgs(args, AgentC51NoisyNetDQNMain.class.getSimpleName());
        run(opts);
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {
        return run(RLRunOptions.defaults());
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {
        var builder = RLConfig.builder()
                .envName("LunarLander-v3")
                .runnerClass(AgentC51NoisyNetDQNMain.class.getSimpleName())
                .algorithmName("c51_noisy_net_dqn")
                .observations(8)
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(1000)
                .batchSize(128)
                .atoms(51)
                .vMin(-100.0f)
                .vMax(+100.0f)
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
                return buildC51NoisyNetDQN(config, env, optimizer, plotTrackers, parent);
            }

            @Override
            public Loss lossFunc() {
                return new CategoricalNLLLoss();
            }
        });
    }

    private static IAgent buildC51NoisyNetDQN(RLConfig config,
                                              IEnv env,
                                              Optimizer optimizer,
                                              PlotTrackers plotTrackers,
                                              NDManager parent) {
        boolean loadModel = config.loadModelPrefix() != null;
        Supplier<IDeepQNetwork> networkFactory = loadModel
                ? () -> new CategoricalQNoisyNetworkMLP(
                        config.observations(),
                        config.actions(),
                        new CategoricalBellmanProjection(
                                config.atoms(), config.vMin(), config.vMax()),
                        config.path(),
                        config.loadModelPrefix(),
                        parent)
                : () -> new CategoricalQNoisyNetworkMLP(
                        config.observations(),
                        config.actions(),
                        config.atoms(),
                        config.vMin(),
                        config.vMax(),
                        parent);

        return new AgentC51NoisyNetDQN(
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
