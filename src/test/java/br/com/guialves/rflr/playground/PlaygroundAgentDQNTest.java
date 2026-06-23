package br.com.guialves.rflr.playground;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.dqn.DeepQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.Gym;
import br.com.guialves.rflr.utils.ExperienceReplayBuffer;
import br.com.guialves.rflr.utils.PlotTrackers;
import org.junit.jupiter.api.Test;

import java.nio.file.Paths;

public class PlaygroundAgentDQNTest {

    @Test
    void shouldTrainAndPlayLunarLander() {

        boolean saveModel = true;
        var path = Paths.get("./");
        try (var manager = NDManager.newBaseManager()) {
            var epsilon = 1.0f;
            var updateQTargetAtTimeN = 1000;
            var minEpsilon = 0.01f;
            var epsilonDecay = 0.995f;
            var gamma = 0.99f;
            var lunarLanderEnv = Gym.make("LunarLander-v3", manager);
            var optimizer = Adam.builder()
                    .optLearningRateTracker(Tracker.fixed(0.0001f))
                    .build(); // or 0.0005

            IO.println("Observation space: " + lunarLanderEnv.observationSpaceStr());
            IO.println("Action space: " + lunarLanderEnv.actionSpaceStr());
            int observations = 8;
            int actions = 4;
            var plotTrackers = new PlotTrackers();

            var agentDQN = new AgentDQN(
                    epsilon,
                    updateQTargetAtTimeN,
                    minEpsilon,
                    epsilonDecay,
                    gamma,
                    lunarLanderEnv,
                    optimizer,
                    () -> new DeepQNetworkMLP(observations, actions, manager),
                    plotTrackers
            );

            int batchSize = 128;
            int framesLimit = 100_000;
            var replayBuffer = new ExperienceReplayBuffer(10_000, new Shape(observations), manager);
            var lossFunc = Loss.l2Loss();
            agentDQN.train(batchSize, framesLimit, replayBuffer, lossFunc);
            if (saveModel) agentDQN.save(path, "LunarLander-v3_" + System.currentTimeMillis() + ".pt");
            double totalReward = agentDQN.run();
            IO.println("Total reward: " + totalReward);
        }
    }
}
