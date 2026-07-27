package br.com.guialves.rflr.execs;

import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.networks.NoisyDuelingQNetworkMLP;
import br.com.guialves.rflr.algorithms.noisydqn.AgentNoisyDuelingNetDDQN;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentNoisyDuelingNetDDQNMain {
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
                .algorithmName("noisy_dueling_nets_ddqn")
                .build();

        RLRunner.run(config, (env, optimizer, plotTrackers) ->
                buildNoisyDuelingNetDDQN(config, env, optimizer, plotTrackers));
    }

    private static IAgent buildNoisyDuelingNetDDQN(RLConfig config,
                                                   IEnv env,
                                                   Optimizer optimizer,
                                                   PlotTrackers plotTrackers) {
        return new AgentNoisyDuelingNetDDQN(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(), // gamma
                env,
                optimizer,
                () -> NoisyDuelingQNetworkMLP.withMeanType(
                    config.observations(),
                    config.actions(),
                    env.manager()
                ),
                plotTrackers
        );
    }
}
