package br.com.guialves.rflr.execs;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import br.com.guialves.rflr.algorithms.IAgent;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.Gym;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import br.com.guialves.rflr.djlutils.DJLUtils;
import br.com.guialves.rflr.algorithms.buffer.ExperienceReplayBuffer;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static br.com.guialves.rflr.djlutils.DJLUtils.gpuCount;

public class RLRunner {

    @FunctionalInterface
    public interface AgentFactory {
        IAgent create(IEnv env, Optimizer optimizer, Device device, PlotTrackers plotTrackers);

        default Loss lossFunc() {
            return Loss.l2Loss();
        }
    }

    public static void run(RLConfig config, AgentFactory agentFactory) {
        var path = Paths.get("./output_models/", config.algorithmName());
        var modelFileName = config.envName() + "_" + System.currentTimeMillis() + "_" + config.algorithmName();

        try (var manager = NDManager.newBaseManager()) {
            var env = Gym.builder()
                    .envName(config.envName())
                    .ndManager(manager)
                    .add(new RecordVideo(path))
                    .add(new RecordEpisodeStatistics())
                    .build();

            var device = gpuCount() > 0 ? Device.gpu() : Device.cpu();
            var optimizer = Adam.builder()
                    .optLearningRateTracker(Tracker.fixed(config.learningRate()))
                    .build();

            IO.println("Engine: " + Engine.getInstance().getEngineName());
            IO.println("GPU count: " + DJLUtils.gpuCount());
            IO.println("Device: " + device);
            IO.println("Observation space: " + env.observationSpaceStr());
            IO.println("Action space: " + env.actionSpaceStr());

            var plotTrackers = new PlotTrackers();

            var agent = agentFactory.create(env, optimizer, device, plotTrackers);

            var replayBuffer = new ExperienceReplayBuffer(config.bufferCapacity(), manager, device);
            var lossFunc = agentFactory.lossFunc();

            agent.train(config.batchSize(), config.framesLimit(), replayBuffer, lossFunc);

            if (config.saveModel()) {
                agent.save(path, modelFileName);
            }

            double totalReward = agent.run();
            IO.println("Info: " + agent.lastInfo());
            IO.println("Replay Buffer size: " + replayBuffer.size());
            IO.println("Total episodes: " + agent.episodes());
            IO.println("Total reward: " + totalReward);
        }

        if (config.saveModel()) {
            validateModelFile(path, modelFileName);
        }
    }

    private static void validateModelFile(Path path, String modelFileName) {
        var fullPath = Paths.get(path.toString(), modelFileName + "-0000.params")
                .normalize().toAbsolutePath();
        if (!Files.exists(fullPath)) {
            throw new IllegalStateException("The model was not generated: " + fullPath);
        }
    }
}
