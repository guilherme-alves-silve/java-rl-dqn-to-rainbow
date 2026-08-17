package br.com.guialves.rflr.execs;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.dqnper.AgentDQNPER;
import br.com.guialves.rflr.algorithms.dqnper.PERL2Loss;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.djlutils.DJLMemoryManagement;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.util.Optional;

import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static br.com.guialves.rflr.utils.PropUtils.getIntProp;

/**
 * Reference:
 *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
 */
public class AgentDQNPERMain {

    static void main() {
        run();
    }

    public static Optional<DJLMemoryManagement.ManagerNode> run() {

        float alpha = 0.4f;
        var config = RLConfig.builder()
                .envName("LunarLander-v3")
                .runnerClass(AgentDQNPERMain.class.getSimpleName())
                .algorithmName("dqnper")
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
                .build();

        return RLRunner.run(config, new RLRunner.AgentFactory() {

            @Override
            public IAgent create(IEnv env, Optimizer optimizer, PlotTrackers plotTrackers, NDManager parent) {
                return buildDQNPER(config, env, optimizer, plotTrackers, parent);
            }

            @Override
            public Loss lossFunc() {
                return PERL2Loss.noneReduction();
            }

            @Override
            public IReplayBuffer replayBuffer(RLConfig config, NDManager manager) {
                return new PrioritizedReplayBuffer(config.bufferCapacity(), alpha, manager);
            }
        });
    }

    private static IAgent buildDQNPER(RLConfig config,
                                      IEnv env,
                                      Optimizer optimizer,
                                      PlotTrackers plotTrackers,
                                      NDManager parent) {
        float initialBeta = 0.6f;
        return new AgentDQNPER(
                config.maxEpsilon(),
                config.updateQTargetAtTimeN(),
                config.minEpsilon(),
                config.epsilonDecay(),
                config.discountFactor(),
                initialBeta,
                env,
                optimizer,
                parent,
                () -> new DeepQNetworkMLP(
                    config.observations(),
                    config.actions(),
                    parent
                ),
                plotTrackers,
                config.debugMemoryLeak()
        );
    }
}
