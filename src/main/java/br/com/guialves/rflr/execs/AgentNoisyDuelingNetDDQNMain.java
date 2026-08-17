package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.networks.NoisyDuelingQNetworkMLP;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.algorithms.noisydqn.AgentNoisyDuelingNetDDQN;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.Optional;

import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;
import static java.lang.System.getProperty;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentNoisyDuelingNetDDQNMain {

    static void main() {
        run();
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {

        var config = RLConfig.builder()
                .envName("LunarLander-v3")
                .runnerClass(AgentNoisyDuelingNetDDQNMain.class.getSimpleName())
                .algorithmName("noisy_dueling_nets_ddqn")
                .observations(8)
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(1000)
                .batchSize(128)
                .duelingType(DuelingType.valueOf(getProperty("agent.duelingType", "MEAN")))
                .framesLimit(getIntProp("agent.framesLimit", "300000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "30000"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "true"))
                .renderRun(getBoolProp("agent.renderRun", "true"))
                .runMaxTries(getIntProp("agent.maxTries", "1"))
                .build();

        return RLRunner.run(config, (env, optimizer, plotTrackers, parent) ->
                buildNoisyDuelingNetDDQN(config, env, optimizer, plotTrackers, parent));
    }

    private static IAgent buildNoisyDuelingNetDDQN(RLConfig config,
                                                   IEnv env,
                                                   Optimizer optimizer,
                                                   PlotTrackers plotTrackers,
                                                   NDManager parent) {
        return new AgentNoisyDuelingNetDDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                parent,
                () -> new NoisyDuelingQNetworkMLP(
                    config.observations(),
                    config.actions(),
                    parent,
                    config.duelingType()
                ),
                plotTrackers,
                config.debugMemoryLeak()
        );
    }
}
