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
import br.com.guialves.rflr.gymnasium4j.wrappers.FrameStackObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.GrayscaleObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.MaxAndSkipObservation;
import br.com.guialves.rflr.gymnasium4j.wrappers.ResizeObservation;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.List;
import java.util.function.Supplier;

import static br.com.guialves.rflr.execs.MainUtils.parseArgs;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * observation_space: Box(0, 255, (224, 256, 3), uint8)
 * action_space: MultiBinary(12)
 * Reference:
 *  <a href="https://stable-retro.farama.org/supported_games/">MegaManX-Snes</a>
 */
public class AgentCNNDQNMain {

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

    /*
    public static List<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {

        var builder = RLConfig.builder()
                .envName("ALE/Galaxian-v5")
                .runnerClass(AgentCNNDQNMain.class.getSimpleName())
                .algorithmName("galaxian_noisy_nets_dqn")
                .observations(84) // 84x84
                .channels(4)
                .actions(6)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(5000)
                .batchSize(32)
                .framesSkip(4)
                .resize(84)
                .concatenate(4)
                .importLibs(List.of("ale_py; gym.register_envs(ale_py)"))
                .framesLimit(getIntProp("agent.framesLimit", "1000000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "500"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "true"))
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
    }*/

    public static List<DJLMemoryManagement.ManagerNode> run(RLRunOptions opts) {

        var builder = RLConfig.builder()
                .envName("ALE/Galaxian-v5")
                .runnerClass(AgentCNNDQNMain.class.getSimpleName())
                .algorithmName("galaxian_noisy_nets_dqn")
                .observations(84) // 84x84
                .channels(4)
                .actions(6)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(5000)
                .batchSize(32)
                .framesSkip(4)
                .resize(84)
                .concatenate(4)
                .importLibs(List.of("ale_py; gym.register_envs(ale_py)"))
                .wrappers(List.of(
                    new MaxAndSkipObservation(4),
                    new GrayscaleObservation(),
                    new ResizeObservation(new int[] {84, 84}),
                    new FrameStackObservation(4)
                ))
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

        return RLRunner.run(config, (env, optimizer, plotTrackers, parent) ->
                buildDQN(config, env, optimizer, plotTrackers, parent));
    }

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
