package br.com.guialves.rflr.execs.mujoco;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkCNN;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.noisydqn.AgentNoisyNetDQN;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.execs.RLConfig;
import br.com.guialves.rflr.execs.RLRunOptions;
import br.com.guialves.rflr.execs.RLRunner;
import br.com.guialves.rflr.execs.RLRunnerCNN;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.Optional;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * observation_space: Box(0, 255, (224, 256, 3), uint8)
 * action_space: MultiBinary(12)
 * Reference:
 *  <a href="https://stable-retro.farama.org/supported_games/"></a>
 */
public class AntAgentNoisyNetDQNMain {

    static void main() {
        run();
    }

    static void main(String[] args) {
        var opts = parseArgs(args, AntAgentNoisyNetDQNMain.class.getSimpleName());
        run(opts);
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {
        return run(RLRunOptions.defaults());
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {
        var builder = RLConfig.builder()
                //.envName("MegaManX-Snes")
                .runnerClass(AntAgentNoisyNetDQNMain.class.getSimpleName())
                .algorithmName("antv5_noisy_nets_dqn")
                .observations(84) // 84x84
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(5000)
                .batchSize(256)
                .framesLimit(getIntProp("agent.framesLimit", "1000000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "30000"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "true"))
                .renderRun(getBoolProp("agent.renderRun", "true"))
                .runMaxTries(getIntProp("agent.maxTries", "1"));

        if (opts.loadModelPrefix() != null) {
            builder = builder.loadModelPrefix(opts.loadModelPrefix());
        }

        var config = builder.build();

        return RLRunnerCNN.run(config, (env, optimizer, plotTrackers, parent) ->
                buildNoisyNetDQN(config, env, optimizer, plotTrackers, parent));
    }

    private static IAgent buildNoisyNetDQN(RLConfig config,
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

        return new AgentNoisyNetDQN(
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
