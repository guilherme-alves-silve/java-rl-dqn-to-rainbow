package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.time.Duration;

import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentDQNMain {
    static void main() throws InterruptedException {

        var config = RLConfig.builder()
                .envName("LunarLander-v3")
                .observations(8)
                .actions(4)
                .learningRate(0.0005f)
                .maxEpsilon(1.0f)
                .minEpsilon(0.01f)
                .discountFactor(0.99f)
                .updateQTargetAtTimeN(1000)
                .batchSize(128)
                .framesLimit(getIntProp("agent.framesLimit", "5000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "500"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .saveModel(true)
                .algorithmName("dqn")
                .build();

        RLRunner.run(config, (env, optimizer, plotTrackers, parent) ->
                buildDQN(config, env, optimizer, plotTrackers, parent));

        //Thread.sleep(Duration.ofMinutes(5));
    }

    private static IAgent buildDQN(RLConfig config,
                                   IEnv env,
                                   Optimizer optimizer,
                                   PlotTrackers plotTrackers,
                                   NDManager parent) {
        return new AgentDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                parent,
                () -> new DeepQNetworkMLP(
                    config.observations(),
                    config.actions(),
                    parent
                ),
                plotTrackers
        );
    }
}
