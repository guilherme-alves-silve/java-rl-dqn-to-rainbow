package br.com.guialves.rflr.execs;

import ai.djl.Device;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentDQNMain {
    static void main() {

        int framesLimit = 200_000;
        float maxEpsilon = 1.0f;
        float minEpsilon = 0.01f;
        float epsilonLinearStep = (maxEpsilon - minEpsilon) / framesLimit;
        float discountFactor = 0.99f;

        var config = new RLConfig(
                "LunarLander-v3", 8, 4,
                0.0005f, maxEpsilon, minEpsilon, epsilonLinearStep, discountFactor,
                1000, 128, framesLimit, 30_000,
                true, "dqn"
        );

        RLRunner.run(config, (env, optimizer, device, plotTrackers) ->
                buildDQN(config, env, optimizer, device, plotTrackers));
    }

    private static IAgent buildDQN(RLConfig config,
                                   IEnv env,
                                   Optimizer optimizer,
                                   Device device,
                                   PlotTrackers plotTrackers) {
        return new AgentDQN(
                config.epsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.gamma(),
                env,
                optimizer,
                () -> new DeepQNetworkMLP(config.observations(),
                        config.actions(),
                        env.manager(),
                        device),
                plotTrackers
        );
    }
}
