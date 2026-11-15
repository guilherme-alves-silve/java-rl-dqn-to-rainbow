package br.com.guialves.rflr.algorithms.buffer;

import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
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
                            @NonNull Function<Integer, IReplayBuffer> iReplayBufferSupplierWithCapacity) {
        this.capacity = capacity;
        this.numberOfEnvs = numberOfEnvs;
        this.capacityPerEnv = getValidCapacityPerEnv();
        this.envNumberAndReplayBuffer = new HashMap<>();
        IntStream.range(0, numberOfEnvs).forEach(i ->
                envNumberAndReplayBuffer.put(i,
                        requireNonNull(iReplayBufferSupplierWithCapacity.apply(capacityPerEnv), "replayBuffer must no be null!")));
    }

    private int getValidCapacityPerEnv() {
        int capacityPerEnv = capacity / numberOfEnvs;
        if (capacity % numberOfEnvs != 0) throw new IllegalArgumentException("The condition (capacity mod numberOfEnvs == 0) must be satisfied!");
        return capacityPerEnv;
    }

    @Override
    public void store(Experience exp) {
        envNumberAndReplayBuffer.get(exp.envNumber()).store(exp);
        if (size < capacity) size += 1;
    }

    @Override
    public boolean enough(int batchSize) {
        return size >= batchSize;
    }

    /**
     *     @Override
     *     public VecExperience sample(int batchSize) {
     *         if (!enough(batchSize)) return null;
     *
     *         var sub = subMgr(subManager, "buffer-sample");
     *         var batch = sampler.sample(experiences, batchSize, size, false);
     *
     *         var states = newAttachedList(sub, batch, exp -> exp.state().duplicate());
     *         var actions = djlMapToLong(sub, batch, exp -> exp.actionAs(Long.class));
     *         var rewards = djlMapToFloat32(sub, batch, Experience::reward);
     *         var nextStates = newAttachedList(sub, batch, exp -> exp.nextState().duplicate());
     *         var dones = djlMapToFloat32(sub, batch, exp -> exp.done() ? 1 : 0);
     *
     *         return new VecExperience(
     *                 sub,
     *                 states,
     *                 actions,
     *                 rewards,
     *                 nextStates,
     *                 dones
     *         );
     *     }
     * @param batchSize The number of samples used to generate the mini-batch.
     * @return
     */

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
}
