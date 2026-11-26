package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.utils.datastructure.RingBuffer;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.List;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.getDebugDump;

public class DJLMemoryLeakChecker {

    /**
     * The maximum number of memory snapshots taken during a training.
     *
     * <p>A single snapshot of the manager tree cannot prove that there is no leak: a leak is
     * a <b>growth</b> over the training. The snapshots are therefore spread over the run, so
     * that the regression test can compare the beginning of the training with its end.</p>
     */
    private static final int MAX_MEMORY_SNAPSHOTS = 100;
    private static final int DEFAULT_MANAGER_NODES = 500;

    private final boolean debugMemoryLeak;
    private final NDManager parent;
    private final RingBuffer<DJLMemoryManagement.ManagerNode> managerNodes;

    public DJLMemoryLeakChecker(boolean debugMemoryLeak, NDManager parent) {
        this.debugMemoryLeak = debugMemoryLeak;
        this.parent = parent;
        this.managerNodes = new RingBuffer<>(DEFAULT_MANAGER_NODES);
    }

    public List<DJLMemoryManagement.ManagerNode> getManagerNodes() {
        return List.copyOf(new ArrayList<>(managerNodes));
    }

    /**
     * The number of steps between two snapshots of the manager tree.
     *
     * <p>The snapshots are spread over the training (at most {@link #MAX_MEMORY_SNAPSHOTS}
     * of them), so that the regression test can compare the beginning of the training with
     * its end: a single snapshot cannot prove that there is no leak, because a leak is a
     * growth over the training.</p>
     *
     * @param framesLimit the number of frames of the training
     * @param framesSkip  the number of frames consumed by one step
     * @return the number of steps between two snapshots (at least one)
     */
    public int snapshotInterval(long framesLimit, long framesSkip) {
        var totalSteps = Math.max(1, framesLimit / Math.max(1, framesSkip));
        return (int) Math.max(1, totalSteps / MAX_MEMORY_SNAPSHOTS);
    }

    /**
     * Offers a snapshot of the manager tree to the {@link #managerNodes}, when the interval
     * of the snapshots is reached and the debug of the memory is enabled.
     *
     * @param step     the number of steps already taken
     * @param interval the number of steps between two snapshots
     */
    public void offerMemorySnapshot(long step, int interval) {
        if (!debugMemoryLeak) return;
        if (step % interval != 0) return;
        getDebugDump(parent).ifPresent(managerNodes::offer);
    }

    public void offerMemorySnapshot() {
        if (!debugMemoryLeak) return;
        getDebugDump(parent).ifPresent(managerNodes::offer);
    }
}
