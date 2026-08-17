package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.duelingdqn.AgentDuelingDQN;
import br.com.guialves.rflr.algorithms.networks.DuelingQNetworkMLP;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
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
public class AgentDuelingDQNMain {

    static void main() {
        run();
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {

        var config = RLConfig.builder()
                .envName("LunarLander-v3")
                .runnerClass(AgentDuelingDQNMain.class.getSimpleName())
                .algorithmName("dueling_dqn")
                .observations(8)
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(1000)
                .batchSize(128)
                .framesLimit(getIntProp("agent.framesLimit", "300000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "30000"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .debugMemoryLeak(getBoolProp("agent.debugMemoryLeak", "true"))
                .renderRun(getBoolProp("agent.renderRun", "true"))
                .runMaxTries(getIntProp("agent.maxTries", "1"))
                .duelingType(DuelingType.valueOf(getProperty("agent.duelingType", "MEAN")))
                .build();

        return RLRunner.run(config, (env, optimizer, plotTrackers, parent) ->
                buildDuelingDQN(config, env, optimizer, plotTrackers, parent));
    }

    private static IAgent buildDuelingDQN(RLConfig config,
                                          IEnv env,
                                          Optimizer optimizer,
                                          PlotTrackers plotTrackers,
                                          NDManager parent) {
        return new AgentDuelingDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                parent,
                () -> new DuelingQNetworkMLP(
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
