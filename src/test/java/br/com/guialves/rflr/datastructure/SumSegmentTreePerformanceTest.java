package br.com.guialves.rflr.datastructure;

import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Performance comparison between SumSegmentTree (node-based) and SumSegmentTreeFlat (array-based).
 * <p>
 * Not a correctness test — validates only that both produce consistent results
 * and reports wall-clock time for each operation at PER-realistic scale.
 * <p>
 * Scenarios:
 *   - update()                   → simulates priority update after TD-error recomputation
 *   - sampleIndexByValueInRange() → simulates mini-batch sampling
 *   - mixed (update + sample)    → simulates a full training step
 */
class SumSegmentTreePerformanceTest {

    private static final int SIZE        = 100_000;
    private static final int OPERATIONS  = 500_000;
    private static final float INIT      = 1.0f;
    private static final long  SEED      = 42L;
    private static final float DELTA     = 1e-6f;

    @Test
    void benchmarkUpdate() {
        var node = new SumSegmentTree(SIZE, INIT);
        var flat = new SumSegmentTreeFlat(SIZE, INIT);
        var rng  = new Random(SEED);

        float[] values = new float[OPERATIONS];
        int[]   idxs   = new int[OPERATIONS];
        for (int i = 0; i < OPERATIONS; i++) {
            idxs[i]   = rng.nextInt(SIZE);
            values[i] = rng.nextFloat() * 10f;
        }

        long nodeTime = timeUpdate(node, idxs, values);
        long flatTime = timeUpdate(flat, idxs, values);

        report("update", nodeTime, flatTime);
    }

    private long timeUpdate(ISumSegmentTree tree, int[] idxs, float[] values) {
        long start = System.nanoTime();
        for (int i = 0; i < idxs.length; i++) tree.update(idxs[i], values[i]);
        return System.nanoTime() - start;
    }

    @Test
    void benchmarkSample() {
        var node = new SumSegmentTree(SIZE, INIT);
        var flat = new SumSegmentTreeFlat(SIZE, INIT);

        // warm up both trees with real priorities
        var rng = new Random(SEED);
        for (int i = 0; i < SIZE; i++) {
            float p = rng.nextFloat() * 10f;
            node.update(i, p);
            flat.update(i, p);
        }

        float nodeDelta = node.sum() * DELTA;
        float flatDelta = flat.sum() * DELTA;

        long nodeTime = timeSample(node, OPERATIONS, nodeDelta);
        long flatTime = timeSample(flat, OPERATIONS, flatDelta);

        report("sampleIndexByValueInRange", nodeTime, flatTime);
    }

    private long timeSample(ISumSegmentTree tree, int ops, float delta) {
        float total = tree.sum();
        long start = System.nanoTime();
        for (int i = 0; i < ops; i++) tree.sampleIndexByValueInRange(delta, total);
        return System.nanoTime() - start;
    }

    @Test
    void benchmarkMixed() {
        var node = new SumSegmentTree(SIZE, INIT);
        var flat = new SumSegmentTreeFlat(SIZE, INIT);

        int batchSize = 32;

        long nodeTime = timeMixed(node, batchSize, OPERATIONS, new Random(SEED));
        long flatTime = timeMixed(flat, batchSize, OPERATIONS, new Random(SEED));

        report("mixed (update + sample, batch=" + batchSize + ")", nodeTime, flatTime);
    }

    private long timeMixed(ISumSegmentTree tree, int batchSize, int steps, Random rng) {
        float delta = tree.sum() * DELTA;
        long start = System.nanoTime();

        for (int step = 0; step < steps; step++) {
            float total = tree.sum();

            // sample a mini-batch
            for (int b = 0; b < batchSize; b++) {
                float segmentWidth = total / batchSize;
                float lower = b * segmentWidth + delta;
                float upper = (b + 1) * segmentWidth;
                if (lower < upper) tree.sampleIndexByValueInRange(lower, upper);
            }

            for (int b = 0; b < batchSize; b++) {
                int idx = rng.nextInt(tree.size());
                tree.update(idx, rng.nextFloat() * 10f);
            }
        }

        return System.nanoTime() - start;
    }

    private void report(String scenario, long nodeNs, long flatNs) {
        double nodeMs = nodeNs / 1_000_000.0;
        double flatMs = flatNs / 1_000_000.0;
        double ratio  = (double) nodeNs / flatNs;

        System.out.printf("%n=== %s ===%n", scenario);
        System.out.printf("  SumSegmentTree (node): %8.2f ms%n", nodeMs);
        System.out.printf("  SumSegmentTreeFlat:    %8.2f ms%n", flatMs);
        System.out.printf("  ratio (node/flat):     %8.2fx  %s%n",
                ratio, ratio > 1.0 ? "→ flat is faster" : "→ node is faster");
    }
}
