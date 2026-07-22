package br.com.guialves.rflr.execs;

import ai.djl.Device;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.dqnper.AgentDQNPER;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentDQNPERMain {
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
                .framesLimit(200_000)
                .bufferCapacity(30_000)
                .saveModel(true)
                .algorithmName("dqnper")
                .build();

        RLRunner.run(config, (env, optimizer, device, plotTrackers) ->
                buildDQNPER(config, env, optimizer, device, plotTrackers));
    }

    private static IAgent<?> buildDQNPER(RLConfig config,
                                         IEnv env,
                                         Optimizer optimizer,
                                         Device device,
                                         PlotTrackers plotTrackers) {
        return new AgentDQNPER(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(),
                env,
                optimizer,
                () -> new DeepQNetworkMLP(
                    config.observations(),
                    config.actions(),
                    env.manager(),
                    device
                ),
                plotTrackers
        );
    }
}
