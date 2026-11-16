package br.com.guialves.rflr.algorithms.buffer;

import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static java.util.Objects.requireNonNull;

public class NEnvReplayBuffer implements IReplayBuffer {

    private final Map<Integer, IReplayBuffer> envNumberAndReplayBuffer;
    private final int capacity;
    private final int numberOfEnvs;
    private final int capacityPerEnv;
    private boolean open;
    private int size;
    private int pos;

    public NEnvReplayBuffer(int capacity,
                            int numberOfEnvs,
                            @NonNull ReplayBufferFactory replayBufferFactory) {
        this.capacity = capacity;
        this.numberOfEnvs = numberOfEnvs;
        this.capacityPerEnv = getValidCapacityPerEnv();
        this.pos = 0;
        var envNumberAndReplayBuffer = new HashMap<Integer, IReplayBuffer>();
        IntStream.range(0, numberOfEnvs).forEach(i ->
                envNumberAndReplayBuffer.put(i,
                        requireNonNull(replayBufferFactory.create(capacityPerEnv), "replayBuffer must no be null!")));
        this.envNumberAndReplayBuffer = Map.copyOf(envNumberAndReplayBuffer);
    }

    private int getValidCapacityPerEnv() {
        int capacityPerEnv = capacity / numberOfEnvs;
        if (capacity % numberOfEnvs != 0) throw new IllegalArgumentException("The condition (capacity mod numberOfEnvs == 0) must be satisfied!");
        return capacityPerEnv;
    }

    @Override
    public void store(Experience exp) {
        envNumberAndReplayBuffer.get(exp.envNumber()).store(exp);
        if (size < capacity) size += capacityPerEnv;
        pos = (pos + capacityPerEnv) % capacity;
    }

    int pos() {
        return pos;
    }

    @Override
    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    @Override
    public IVecExperience sample(int batchSize) {
        if (!enough(batchSize)) return null;

        int batchSizePerEnv = batchSize / numberOfEnvs;
        if (batchSize % numberOfEnvs != 0) throw new IllegalArgumentException("The condition (batchSize mod numberOfEnvs == 0) must be satisfied!");
        // TODO: Finish
        for (var replayBuffer : envNumberAndReplayBuffer.values()) {
            var vecExperience = replayBuffer.sample(batchSizePerEnv);
            //vecExperience.
        }

        return null;
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
        return open;
    }

    @Override
    public void close() {
        this.open = false;
        this.envNumberAndReplayBuffer.forEach((_, replayBuffer) -> replayBuffer.close());
    }

    @FunctionalInterface
    public interface ReplayBufferFactory {
        IReplayBuffer create(int capacity);
    }
}
