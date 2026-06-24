package br.com.guialves.rflr.execs;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import br.com.guialves.rflr.algorithms.dqn.AgentDQN;
import br.com.guialves.rflr.algorithms.networks.DeepQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.Gym;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import br.com.guialves.rflr.utils.DJLUtils;
import br.com.guialves.rflr.utils.ExperienceReplayBuffer;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.nio.file.Files;
import java.nio.file.Paths;

import static br.com.guialves.rflr.utils.DJLUtils.gpuCount;

public class AgentDQNMain {

    /**
     * Reference:
     *  <a href="https://gymnasium.farama.org/environments/box2d/lunar_lander/">Lunar Lander</a>
     */
    static void main() {

        boolean saveModel = true;
        var path = Paths.get("./output_models/dqn/");
        var modelFileName = "LunarLander-v3_" + System.currentTimeMillis() + "_dqn";
        try (var manager = NDManager.newBaseManager()) {
            var epsilon = 1.0f;
            var updateQTargetAtTimeN = 1000;
            var minEpsilon = 0.01f;
            var epsilonDecay = 0.995f;
            var gamma = 0.99f;
            var lunarLanderEnv = Gym.builder()
                    .envName("LunarLander-v3")
                    .ndManager(manager)
                    .add(new RecordVideo(path))
                    .add(new RecordEpisodeStatistics())
                    .build();
            var device = gpuCount() > 0? Device.gpu() : Device.cpu();
            var optimizer = Adam.builder()
                    .optLearningRateTracker(Tracker.fixed(0.0005f))
                    .build();

            IO.println("Engine: " + Engine.getInstance().getEngineName());
            IO.println("GPU count: " + DJLUtils.gpuCount());
            IO.println("Device: " + device);
            IO.println("Observation space: " + lunarLanderEnv.observationSpaceStr());
            IO.println("Action space: " + lunarLanderEnv.actionSpaceStr());
            int observations = 8;
            int actions = 4;
            var plotTrackers = new PlotTrackers();
            int batchSize = 128;
            int framesLimit = 100_000;
            int bufferCapacity = 30_000;

            var agentDQN = new AgentDQN(
                    epsilon,
                    updateQTargetAtTimeN,
                    minEpsilon,
                    epsilonDecay,
                    gamma,
                    lunarLanderEnv,
                    device,
                    optimizer,
                    () -> new DeepQNetworkMLP(observations, actions, manager, device),
                    plotTrackers
            );

            var replayBuffer = new ExperienceReplayBuffer(bufferCapacity, manager, device);
            var lossFunc = Loss.l2Loss();
            agentDQN.train(batchSize, framesLimit, replayBuffer, lossFunc);
            if (saveModel) agentDQN.save(path, modelFileName);
            double totalReward = agentDQN.run();
            IO.println("Info: " + agentDQN.lastInfo());
            IO.println("Replay Buffer size: " + replayBuffer.size());
            IO.println("Total episodes: " + agentDQN.episodes());
            IO.println("Total reward: " + totalReward);
        }

        var fullPath = Paths.get(path.toString(), modelFileName + "-0000.params")
                .normalize().toAbsolutePath();
        if (!Files.exists(fullPath)) {
            throw new IllegalStateException("The model was not generated: " + fullPath);
        }
    }
}
