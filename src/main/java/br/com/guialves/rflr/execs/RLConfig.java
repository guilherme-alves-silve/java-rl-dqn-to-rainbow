package br.com.guialves.rflr.execs;

import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import lombok.Builder;
import lombok.NonNull;
import lombok.SneakyThrows;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Optional;

import static br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer.DEFAULT_ALPHA;
import static br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer.DEFAULT_BETA;
import static br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection.*;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.ManagerNode;

@Builder(toBuilder = true)
public record RLConfig(
    String envName,
    String runnerClass,
    String algorithmName,
    int observations,
    int actions,
    float learningRate,
    float maxEpsilon,
    float minEpsilon,
    float epsilonDecay,
    float discountFactor, // or gamma
    float alpha, // for PER
    float beta, // for PER
    int updateQTargetAtTimeN,
    int batchSize,
    int nStep,
    int framesLimit,
    int bufferCapacity,
    int atoms,
    float vMin,
    float vMax,
    boolean saveModel,
    DuelingType duelingType,
    Path path,
    boolean debugMemoryLeak,
    int runMaxTries,
    boolean renderRun,
    String loadModelPrefix
) {

    private static final int DEFAULT_N_STEP = 1;
    private static final float DELTA = 1e-6f;

    public RLConfig {
        epsilonDecay = (maxEpsilon - minEpsilon) / framesLimit;
        path = Paths.get("./output_models/", algorithmName);
    }

    @Override
    public float vMin() {
        if (vMin <= DELTA) return V_MIN;
        return vMin;
    }

    @Override
    public float vMax() {
        if (vMax <= DELTA) return V_MAX;
        return vMax;
    }

    @Override
    public int atoms() {
        if (atoms <= 0) return N_ATOMS;
        return atoms;
    }

    @Override
    public int nStep() {
        if (nStep <= 0) return DEFAULT_N_STEP;
        return nStep;
    }

    @Override
    public float beta() {
        if (beta <= DELTA) return DEFAULT_BETA;
        return beta;
    }

    @Override
    public float alpha() {
        if (alpha <= DELTA) return DEFAULT_ALPHA;
        return alpha;
    }

    public RecordVideo recordVideo() {
        return new RecordVideo(path);
    }

    public RecordEpisodeStatistics recordEpisodeStatistics() {
        return new RecordEpisodeStatistics();
    }

    @SneakyThrows
    public void saveConfig(@NonNull Optional<ManagerNode> managerNode, Map<String, Object> runOutput) {
        var strManagerNode = managerNode.map(ManagerNode::toString).orElse("null");
        var outputPath = Paths.get("./docs/training_results/%s/%s.txt".formatted(algorithmName, runnerClass));
        Files.createDirectories(outputPath.getParent());
        Files.writeString(outputPath, this + System.lineSeparator() + strManagerNode +
                        System.lineSeparator() + runOutput.toString(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE);
    }
}
