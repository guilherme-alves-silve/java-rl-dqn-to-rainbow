package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.Accessors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static java.util.Objects.requireNonNull;

@Accessors(fluent = true)
public class NEnvReplayBuffer implements IReplayBuffer {

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

    @Override
    public boolean enough(int batchSize) {
        if (batchSize % numberOfEnvs != 0) return false;
        int batchSizePerEnv = batchSize / numberOfEnvs;
        // Avoiding lambda to make sure it runs faster
        for (var buffer : envNumberAndReplayBuffer.values()) {
            if (!buffer.enough(batchSizePerEnv)) {
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

        int batchSizePerEnv = batchSize / numberOfEnvs;
        if (batchSize % numberOfEnvs != 0) throw new IllegalArgumentException("The condition (batchSize mod numberOfEnvs == 0) must be satisfied!");
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
                var vecExperience = replayBuffer.sample(batchSizePerEnv);
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
                    NDArrays.concat(listStates),
                    NDArrays.concat(listActions),
                    NDArrays.concat(listRewards),
                    NDArrays.concat(listNextStates),
                    NDArrays.concat(listDones)
            );
        } finally {
            samplesPerEnvToClose.forEach(IVecExperience::close);
        }
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
