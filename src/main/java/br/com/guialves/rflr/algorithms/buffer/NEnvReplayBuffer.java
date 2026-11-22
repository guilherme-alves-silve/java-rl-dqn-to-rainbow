package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.IntStream;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static java.util.Collections.*;
import static java.util.Objects.requireNonNull;

/**
 * A replay buffer that holds one buffer per environment.
 *
 * <p>Every environment has its own buffer, so the experience of an environment is never
 * mixed with the experience of another one. The batches are assembled by sampling each
 * environment and concatenating the results, in environment order.</p>
 *
 * <p>It supports the Prioritized Experience Replay: the {@link ReplayBufferFactory} must
 * create a {@link IPrioritizedBuffer} per environment, and the buffer indexes of the
 * concatenated batch are globalized as {@code envNumber * capacityPerEnv + localIndex}, so
 * that {@link #updatePriorities(int[], NDArray)} can route each priority back to the buffer
 * that owns the sample.</p>
 *
 * @see IPrioritizedBuffer
 */
@Slf4j
@Accessors(fluent = true)
public class NEnvReplayBuffer implements IPrioritizedBuffer, INStepBuffer {

    private final Map<Integer, IReplayBuffer> envNumberAndReplayBuffer;
    private final int capacity;
    @Getter
    private final int numberOfEnvs;
    private final int capacityPerEnv;
    private final NDManager subManager;
    private int size;

    public NEnvReplayBuffer(int capacity,
                            int numberOfEnvs,
                            NDManager parent,
                            @NonNull ReplayBufferFactory replayBufferFactory) {
        this.capacity = capacity;
        this.numberOfEnvs = numberOfEnvs;
        this.capacityPerEnv = getValidCapacityPerEnv();
        // this sub manager stays in the GPU (if available)
        this.subManager = subMgr(parent, getClass());
        var envNumberAndReplayBuffer = new HashMap<Integer, IReplayBuffer>();
        IntStream.range(0, numberOfEnvs).forEach(i ->
                envNumberAndReplayBuffer.put(i,
                        requireNonNull(replayBufferFactory.create(capacityPerEnv), "replayBuffer must no be null!")));
        // immutable hash-map
        this.envNumberAndReplayBuffer = Map.copyOf(envNumberAndReplayBuffer);

        warnAboutNStepBuffersWithDifferentNSteps(envNumberAndReplayBuffer);
    }

    private static void warnAboutNStepBuffersWithDifferentNSteps(HashMap<Integer, IReplayBuffer> envNumberAndReplayBuffer) {
        var nStepsSizes = new ArrayList<Integer>();
        for (var buffer : envNumberAndReplayBuffer.values()) {
            if (buffer instanceof INStepBuffer nStepBuffer) {
                nStepsSizes.add(nStepBuffer.nStep());
            }
        }

        if (nStepsSizes.isEmpty()) return;

        if (!Objects.equals(min(nStepsSizes), max(nStepsSizes))) {
            log.warn("Different nStep between the buffers! Are you sure? {}", nStepsSizes);
        }
    }

    private int getValidCapacityPerEnv() {
        int capacityPerEnv = capacity / numberOfEnvs;
        if (capacity % numberOfEnvs != 0) throw new IllegalArgumentException("The condition (capacity mod numberOfEnvs == 0) must be satisfied!");
        return capacityPerEnv;
    }

    @Override
    public void store(Experience exp) {
        var replayBuffer = envNumberAndReplayBuffer.get(exp.envNumber());
        if (replayBuffer == null) {
            throw new IllegalArgumentException("Invalid env number: " + exp.envNumber()
                    + " (expected 0.." + (numberOfEnvs - 1) + ")");
        }
        replayBuffer.store(exp);
        if (size < capacity) size += 1;
        size = getSize();
    }

    /**
     * The number of samples that each environment contributes to a batch.
     *
     * <p>The batch is assembled in environment order and every environment
     * contributes at least one sample, so that no environment is left out of the
     * batch. When the batch size is not a multiple of the number of environments,
     * the remainder is spread over the first environments.</p>
     *
     * @param batchSize the size of the whole batch
     * @return an array with the number of samples of each environment
     */
    private int[] samplesPerEnv(int batchSize) {
        if (batchSize < numberOfEnvs) {
            throw new IllegalArgumentException(
                    ("The batch size (%d) must be at least the number of environments (%d)," +
                     "otherwise some environment would not be part of the batch!").formatted(batchSize, numberOfEnvs));
        }

        int[] samplesPerEnv = new int[numberOfEnvs];
        int base = batchSize / numberOfEnvs;
        int remainder = batchSize % numberOfEnvs;
        for (int idxEnv = 0; idxEnv < numberOfEnvs; ++idxEnv) {
            samplesPerEnv[idxEnv] = base + (idxEnv < remainder ? 1 : 0);
        }

        return samplesPerEnv;
    }

    @Override
    public boolean enough(int batchSize) {
        int[] samplesPerEnv = samplesPerEnv(batchSize);
        // Avoiding lambda to make sure it runs faster
        for (int envNumber = 0;  envNumber < numberOfEnvs; ++envNumber) {
            var buffer = envNumberAndReplayBuffer.get(envNumber);
            if (!buffer.enough(samplesPerEnv[envNumber])) {
                return false;
            }
        }
        return true;
    }

    private int getSize() {
        // Avoiding lambda to make sure it runs faster
        int sum = 0;
        for (var env : envNumberAndReplayBuffer.values()) {
            sum += env.size();
        }
        return sum;
    }

    @Override
    public IVecExperience sample(int batchSize) {
        if (!enough(batchSize)) return null;

        int[] samplesPerEnv = samplesPerEnv(batchSize);
        var sub = subMgr(subManager, "n-buffer-sample");
        var listStates = new NDList(numberOfEnvs);
        var listActions = new NDList(numberOfEnvs);
        var listRewards = new NDList(numberOfEnvs);
        var listNextStates = new NDList(numberOfEnvs);
        var listDones = new NDList(numberOfEnvs);

        var samplesPerEnvToClose = new ArrayList<IVecExperience>(numberOfEnvs);
        try {
            for (int envNumber = 0; envNumber < numberOfEnvs; ++envNumber) {
                var replayBuffer = envNumberAndReplayBuffer.get(envNumber);
                var vecExperience = replayBuffer.sample(samplesPerEnv[envNumber]);
                samplesPerEnvToClose.add(vecExperience);
                listStates.add(vecExperience.states());
                listActions.add(vecExperience.actions());
                listRewards.add(vecExperience.rewards());
                listNextStates.add(vecExperience.nextStates());
                listDones.add(vecExperience.dones());
            }

            sub.attachAll(listStates, listActions, listRewards, listNextStates, listDones);

            return new ExperienceReplayBuffer.VecExperience(
                    sub,
                    concat(sub, listStates),
                    concat(sub, listActions),
                    concat(sub, listRewards),
                    concat(sub, listNextStates),
                    concat(sub, listDones)
            );
        } finally {
            samplesPerEnvToClose.forEach(IVecExperience::close);
        }
    }

    /**
     * Samples a prioritized mini-batch: each environment is sampled with its own
     * priorities and the results are concatenated in environment order.
     *
     * @param batchSize the number of samples of the whole batch
     * @param beta      the degree of the importance-sampling correction
     * @return the batch with the importance-sampling weights and the globalized indexes
     */
    @Override
    public PrioritizedVecExperience sample(int batchSize, float beta) {
        if (!enough(batchSize)) return null;

        int[] samplesPerEnv = samplesPerEnv(batchSize);
        var sub = subMgr(subManager, "n-buffer-sample");
        var listStates = new NDList(numberOfEnvs);
        var listActions = new NDList(numberOfEnvs);
        var listRewards = new NDList(numberOfEnvs);
        var listNextStates = new NDList(numberOfEnvs);
        var listDones = new NDList(numberOfEnvs);
        var listWeights = new NDList(numberOfEnvs);
        var globalIndexes = new int[batchSize];

        var samplesPerEnvToClose = new ArrayList<IVecExperience>(numberOfEnvs);
        try {
            int offset = 0;
            for (int envNumber = 0; envNumber < numberOfEnvs; ++envNumber) {
                var replayBuffer = envNumberAndReplayBuffer.get(envNumber);
                if (!(replayBuffer instanceof IPrioritizedBuffer prioritizedBuffer)) {
                    throw new IllegalArgumentException("The buffer of the environment %d is not a prioritized buffer!"
                            .formatted(envNumber));
                }

                var vecExperience = prioritizedBuffer.sample(samplesPerEnv[envNumber], beta);
                samplesPerEnvToClose.add(vecExperience);
                listStates.add(vecExperience.states());
                listActions.add(vecExperience.actions());
                listRewards.add(vecExperience.rewards());
                listNextStates.add(vecExperience.nextStates());
                listDones.add(vecExperience.dones());
                listWeights.add(vecExperience.weights());

                var localIndexes = vecExperience.bufferIndexes();
                for (int i = 0; i < localIndexes.length; ++i) {
                    globalIndexes[offset + i] = envNumber * capacityPerEnv + localIndexes[i];
                }
                offset += samplesPerEnv[envNumber];
            }

            sub.attachAll(listStates, listActions, listRewards, listNextStates, listDones);

            return new PrioritizedVecExperience(
                    sub,
                    concat(sub, listStates),
                    concat(sub, listActions),
                    concat(sub, listRewards),
                    concat(sub, listNextStates),
                    concat(sub, listDones),
                    concat(sub, listWeights),
                    globalIndexes
            );
        } finally {
            samplesPerEnvToClose.forEach(IVecExperience::close);
        }
    }

    /**
     * Updates the priorities of the samples, routing each one to the buffer of the
     * environment that owns it.
     *
     * @param bufferIndexes the globalized indexes, as returned by
     *                       {@link #sample(int, float)}
     * @param priorities    the new priorities, one per sample
     */
    @Override
    public void updatePriorities(int[] bufferIndexes, NDArray priorities) {
        if (bufferIndexes.length != priorities.size()) throw new IllegalArgumentException("Invalid length!");
        var samplesPerEnv = samplesPerEnv(bufferIndexes.length);
        int offset = 0;
        for (int envNumber = 0; envNumber < numberOfEnvs; ++envNumber) {
            var replayBuffer = envNumberAndReplayBuffer.get(envNumber);
            if (!(replayBuffer instanceof IPrioritizedBuffer prioritizedBuffer)) {
                throw new IllegalArgumentException("The buffer of the environment %d is not a prioritized buffer!"
                        .formatted(envNumber));
            }

            int count = samplesPerEnv[envNumber];
            var localIndexes = new int[count];
            for (int i = 0; i < count; ++i) {
                var globalIndex = bufferIndexes[offset + i];
                localIndexes[i] = globalIndex - envNumber * capacityPerEnv;
            }
            var localPriorities = priorities.get(NDIndex.sliceAxis(0, offset, offset + count));
            prioritizedBuffer.updatePriorities(localIndexes, localPriorities);
            offset += count;
        }
    }

    private static NDArray concat(NDManager sub, NDList arrays) {
        var concat = NDArrays.concat(arrays);
        concat.attach(sub);
        return concat;
    }

    /**
     * @return the n-step of the buffers of the environments
     */
    @Override
    public int nStep() {
        for (var buffer : envNumberAndReplayBuffer.values()) {
            if (buffer instanceof INStepBuffer nStepBuffer) {
                return nStepBuffer.nStep();
            }
        }

        return 1;
    }

    int capacityPerEnv() {
        return capacityPerEnv;
    }

    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isOpen() {
        return subManager.isOpen();
    }

    @Override
    public void close() {
        this.envNumberAndReplayBuffer.forEach((_, replayBuffer) -> replayBuffer.close());
        subManager.close();
    }

    @FunctionalInterface
    public interface ReplayBufferFactory {
        IReplayBuffer create(int capacity);
    }

    public record VecExperience(NDManager sub,
                                NDArray states,
                                NDArray actions,
                                NDArray rewards,
                                NDArray nextStates,
                                NDArray dones) implements IVecExperience {
        @Override
        public void close() {
            sub.close();
        }
    }
}
