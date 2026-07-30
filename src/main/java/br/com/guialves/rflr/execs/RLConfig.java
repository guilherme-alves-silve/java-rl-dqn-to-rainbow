package br.com.guialves.rflr.execs;

import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import lombok.Builder;

import java.nio.file.Path;
import java.nio.file.Paths;

@Builder(toBuilder = true)
public record RLConfig(
    String envName,
    int observations,
    int actions,
    float learningRate,
    float maxEpsilon,
    float minEpsilon,
    float epsilonDecay,
    float discountFactor, // or gamma
    int updateQTargetAtTimeN,
    int batchSize,
    int nStep,
    int framesLimit,
    int bufferCapacity,
    boolean saveModel,
    String algorithmName,
    DuelingType duelingType,
    Path path,
    boolean debugMemoryLeak,
    int runMaxTries,
    boolean renderRun
) {
    public RLConfig {
        epsilonDecay = (maxEpsilon - minEpsilon) / framesLimit;
        path = Paths.get("./output_models/", algorithmName);
    }

    public RecordVideo recordVideo() {
        return new RecordVideo(path);
    }

    public RecordEpisodeStatistics recordEpisodeStatistics() {
        return new RecordEpisodeStatistics();
    }
}
