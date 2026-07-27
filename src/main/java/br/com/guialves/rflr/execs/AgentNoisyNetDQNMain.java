package br.com.guialves.rflr.execs;

import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.algorithms.networks.NoisyNetworkMLP;
import br.com.guialves.rflr.algorithms.noisydqn.AgentNoisyNetDQN;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentNoisyNetDQNMain {
    static void main() {

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
                .framesLimit(getIntProp("agent.framesLimit", "300000"))
                .bufferCapacity(getIntProp("agent.bufferCapacity", "30000"))
                .saveModel(getBoolProp("agent.saveModel", "true"))
                .saveModel(true)
                .algorithmName("dqn")
                .build();

        RLRunner.run(config, (env, optimizer, plotTrackers) ->
                buildNoisyNetDQN(config, env, optimizer, plotTrackers));
    }

    private static IAgent buildNoisyNetDQN(RLConfig config,
                                           IEnv env,
                                           Optimizer optimizer,
                                           PlotTrackers plotTrackers) {
        return new AgentNoisyNetDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                () -> new NoisyNetworkMLP(
                    config.observations(),
                    config.actions(),
                    env.manager()
                ),
                plotTrackers
        );
    }
}
