package br.com.guialves.rflr.execs;

import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.gymnasium4j.Gym;
import br.com.guialves.rflr.gymnasium4j.wrappers.IWrapper;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordEpisodeStatistics;
import br.com.guialves.rflr.gymnasium4j.wrappers.RecordVideo;
import br.com.guialves.rflr.utils.ModelFileNames;
import lombok.Builder;
import lombok.SneakyThrows;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer.DEFAULT_ALPHA;
import static br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer.DEFAULT_BETA;
import static br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection.*;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.ManagerNode;
import static java.util.stream.Collectors.joining;

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
    String loadModelPrefix,
    Vectorized vectorized,
    int channels,
    boolean stableRetro,
    boolean vectorizedEnv,
    Gym.PyMap envParams,
    List<String> importLibs,
    List<IWrapper> wrappers,
    int framesSkip,
    int resize,
    int concatenate
) {

    private static final int DEFAULT_N_STEP = 1;
    private static final int DEFAULT_CHANNELS = 1;
    private static final float DELTA = 1e-6f;
    private static final int DEFAULT_FRAMES_SKIP = 1;

    public enum Vectorized {
        SYNC(false), ASYNC(true), NONE(false);

        private final boolean async;

        Vectorized(boolean async) {
            this.async = async;
        }

        public boolean isAsync() {
            return async;
        }
    }

    public RLConfig {
        framesSkip = framesSkip <= 0 ? DEFAULT_FRAMES_SKIP : framesSkip;
        epsilonDecay = epsilonDecay > 0 ? epsilonDecay : (maxEpsilon - minEpsilon) / framesLimit;
        path = path != null ? path : Paths.get("./output_models/", algorithmName);
        vectorized = vectorized == null ? Vectorized.NONE : vectorized;
        channels = channels <= 0 ? DEFAULT_CHANNELS : channels;
        envParams = envParams == null ? Gym.builderMap() : envParams;
    }

    public void importLibsIn(Gym.EnvBuilder envBuilder) {
        importLibs().forEach(envBuilder::importLib);
    }

    /**
     * Name of the file that holds the model trained by this configuration, without
     * the {@code -NNNN.params} suffix added by DJL.
     * <p>
     * The environment name may contain a path separator (e.g. {@code ALE/Galaxian-v5}),
     * so it is sanitized to keep every model in a single folder.
     *
     * @return the model file name used when saving and when reloading the model
     */
    public String modelFileName() {
        return ModelFileNames.build(envName(), algorithmName());
    }

    @Override
    public List<String> importLibs() {
        if (null == importLibs) return Collections.emptyList();
        return importLibs;
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

    @Override
    public List<IWrapper> wrappers() {
        if (null == wrappers) return List.of();
        return wrappers;
    }

    public RecordVideo recordVideo() {
        return new RecordVideo(path);
    }

    public RecordEpisodeStatistics recordEpisodeStatistics() {
        return new RecordEpisodeStatistics();
    }

    @SneakyThrows
    public void saveConfig(List<ManagerNode> managerNodes, Map<String, Object> runOutput) {
        var jsonManagerNodes = managerNodes.stream()
                .map(ManagerNode::toString)
                .collect(joining(",", "[", "]"));
        var outputPath = Paths.get("./docs/training_results/%s/%s.txt".formatted(algorithmName, runnerClass));
        Files.createDirectories(outputPath.getParent());
        Files.writeString(outputPath, this + System.lineSeparator() + jsonManagerNodes +
                        System.lineSeparator() + runOutput.toString(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE);
    }
}
