package br.com.guialves.rflr.datastructure;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

class SumSegmentTreeTest extends SumSegmentTreeContractTest {
    @Override
    ISumSegmentTree create(int size, float initValue) {
        return new SumSegmentTree(size, initValue);
    }
}

class SumSegmentTreeFlatTest extends SumSegmentTreeContractTest {
    @Override
    ISumSegmentTree create(int size, float initValue) {
        return new SumSegmentTreeFlat(size, initValue);
    }
}

abstract class SumSegmentTreeContractTest {

    private static final float DELTA = 1e-5f;

    // [1, 2, 3, 4, 5] -> sum = 15
    private ISumSegmentTree tree5;
    // [1, 2, 3, 4, 5, 6, 7, 8] -> sum = 36
    private ISumSegmentTree tree8;

    abstract ISumSegmentTree create(int size, float initValue);

    private ISumSegmentTree fromArray(float... values) {
        var tree = create(values.length, 0f);
        for (int i = 0; i < values.length; i++) tree.update(i, values[i]);
        return tree;
    }

    @BeforeEach
    void setUp() {
        tree5 = fromArray(1, 2, 3, 4, 5);
        tree8 = fromArray(1, 2, 3, 4, 5, 6, 7, 8);
    }

    // =========================================================
    // Constructor
    // =========================================================

    @Test
    void testConstructorSizeAndInitValue() {
        var tree = create(5, 2f);
        assertEquals(5, tree.size());
        assertEquals(10f, tree.sum(), DELTA); // 5 * 2 = 10
    }

    @Test
    void testConstructorZeroInitValue() {
        var tree = create(4, 0f);
        assertEquals(4, tree.size());
        assertEquals(0f, tree.sum(), DELTA);
    }

    @Test
    void testConstructorSingleElement() {
        var tree = create(1, 10f);
        assertEquals(1, tree.size());
        assertEquals(10f, tree.rangeSum(0, 0), DELTA);
    }

    // =========================================================
    // sum()
    // =========================================================

    @Test
    void testSumReturnsTotal() {
        assertEquals(15f, tree5.sum(), DELTA);
        assertEquals(36f, tree8.sum(), DELTA);
    }

    @Test
    void testSumUpdatesAfterUpdate() {
        tree5.update(2, 10f);
        assertEquals(22f, tree5.sum(), DELTA);
    }

    // =========================================================
    // update()
    // =========================================================

    @Test
    void testUpdateMiddleElement() {
        tree5.update(2, 10f);
        assertEquals(22f, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testUpdateFirstElement() {
        tree5.update(0, 100f);
        assertEquals(100f + 2 + 3 + 4 + 5, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testUpdateLastElement() {
        tree5.update(4, 50f);
        assertEquals(1 + 2 + 3 + 4 + 50f, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testMultipleUpdates() {
        tree5.update(1, 20f);
        tree5.update(3, 30f);
        assertEquals(1 + 20f + 3 + 30f + 5, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testUpdateSameIndexMultipleTimes() {
        tree5.update(2, 10f);
        assertEquals(1 + 2 + 10f + 4 + 5, tree5.rangeSum(0, 4), DELTA);

        tree5.update(2, 20f);
        assertEquals(1 + 2 + 20f + 4 + 5, tree5.rangeSum(0, 4), DELTA);

        tree5.update(2, 3f);
        assertEquals(15f, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testUpdateOutOfBoundsIsIgnored() {
        assertDoesNotThrow(() -> tree5.update(-1, 10f));
        assertDoesNotThrow(() -> tree5.update(10, 10f));
        assertEquals(15f, tree5.sum(), DELTA);
    }

    // =========================================================
    // rangeSum()
    // =========================================================

    @Test
    void testRangeSumFullArray() {
        assertEquals(36f, tree8.rangeSum(0, 7), DELTA);
    }

    @Test
    void testRangeSumSingleElement() {
        assertEquals(1f, tree8.rangeSum(0, 0), DELTA);
        assertEquals(5f, tree8.rangeSum(4, 4), DELTA);
        assertEquals(8f, tree8.rangeSum(7, 7), DELTA);
    }

    @Test
    void testRangeSumMiddleSubarray() {
        assertEquals(12f, tree8.rangeSum(2, 4), DELTA);
        assertEquals(9f,  tree8.rangeSum(1, 3), DELTA);
    }

    @Test
    void testRangeSumPrefix() {
        assertEquals(1f,  tree8.rangeSum(0, 0), DELTA);
        assertEquals(3f,  tree8.rangeSum(0, 1), DELTA);
        assertEquals(6f,  tree8.rangeSum(0, 2), DELTA);
        assertEquals(10f, tree8.rangeSum(0, 3), DELTA);
    }

    @Test
    void testRangeSumSuffix() {
        assertEquals(8f,  tree8.rangeSum(7, 7), DELTA);
        assertEquals(15f, tree8.rangeSum(6, 7), DELTA);
        assertEquals(21f, tree8.rangeSum(5, 7), DELTA);
        assertEquals(26f, tree8.rangeSum(4, 7), DELTA);
    }

    @Test
    void testRangeSumInvalidOrder() {
        assertEquals(0f, tree8.rangeSum(5, 3), DELTA);
        assertEquals(0f, tree8.rangeSum(7, 0), DELTA);
    }

    @Test
    void testRangeSumOutOfBounds() {
        assertEquals(0f, tree8.rangeSum(-5, -1), DELTA);
        assertEquals(0f, tree8.rangeSum(10, 15), DELTA);
        assertEquals(1f, tree8.rangeSum(-1, 0),  DELTA);
        assertEquals(8f, tree8.rangeSum(7, 10),  DELTA);
    }

    @Test
    void testRangeSumAfterUpdates() {
        assertEquals(6f, tree8.rangeSum(0, 2), DELTA);

        tree8.update(1, 10f);
        assertEquals(14f, tree8.rangeSum(0, 2), DELTA);

        tree8.update(5, 20f);
        assertEquals(32f, tree8.rangeSum(2, 5), DELTA);

        tree8.update(0, 100f);
        assertEquals(157f, tree8.rangeSum(0, 7), DELTA);
    }

    @ParameterizedTest
    @CsvSource({
            "0, 0, 1",
            "0, 1, 3",
            "0, 2, 6",
            "1, 3, 9",
            "2, 5, 18",
            "3, 6, 22",
            "4, 7, 26"
    })
    void testRangeSumParameterized(int left, int right, float expected) {
        assertEquals(expected, tree8.rangeSum(left, right), DELTA);
    }

    // =========================================================
    // prefixSum()
    // =========================================================

    @Test
    void testPrefixSumDefinitionConsistency() {
        float[] array = {3, 1, 4, 1, 5};
        var tree = fromArray(array);

        float running = 0;
        for (int i = 0; i < array.length; i++) {
            running += array[i];
            assertEquals(running, tree.prefixSum(i), DELTA);
        }
    }

    @Test
    void testPrefixSumAfterMultipleUpdates() {
        float[] array = {1, 2, 3, 4, 5};
        var tree = fromArray(array);

        tree.update(0, 10f);
        tree.update(3, 20f);
        array[0] = 10f;
        array[3] = 20f;

        float running = 0;
        for (int i = 0; i < array.length; i++) {
            running += array[i];
            assertEquals(running, tree.prefixSum(i), DELTA);
        }
    }

    @Test
    void testPrefixSumEquivalentToNumpyCumsum() {
        var tree = fromArray(5, 3, 8, 2, 7);
        float[] cumsum = {5, 8, 16, 18, 25};
        for (int i = 0; i < cumsum.length; i++)
            assertEquals(cumsum[i], tree.prefixSum(i), DELTA);
    }

    // =========================================================
    // sampleIndexByValueInRange() — deterministic via narrow range
    // =========================================================

    @Test
    void testUniformArray() {
        // [2, 2, 2, 2] -> prefix sums [2, 4, 6, 8]
        var tree = fromArray(2, 2, 2, 2);
        for (int i = 0; i < 50; i++) {
            assertEquals(0, tree.sampleIndexByValueInRange(DELTA,        2.0f));
            assertEquals(1, tree.sampleIndexByValueInRange(2.0f + DELTA, 4.0f));
            assertEquals(2, tree.sampleIndexByValueInRange(4.0f + DELTA, 6.0f));
            assertEquals(3, tree.sampleIndexByValueInRange(6.0f + DELTA, 8.0f));
        }
    }

    @Test
    void testNonUniformArray() {
        // [1, 3, 2, 4] -> prefix sums [1, 4, 6, 10]
        var tree = fromArray(1, 3, 2, 4);
        for (int i = 0; i < 50; i++) {
            assertEquals(0, tree.sampleIndexByValueInRange(DELTA,        1.0f));
            assertEquals(1, tree.sampleIndexByValueInRange(1.0f + DELTA, 4.0f));
            assertEquals(2, tree.sampleIndexByValueInRange(4.0f + DELTA, 6.0f));
            assertEquals(3, tree.sampleIndexByValueInRange(6.0f + DELTA, 10.0f));
        }
    }

    @Test
    void testSampleAfterUpdateSlotShifts() {
        // [1, 3, 2, 4] -> update(1, 7f) -> prefix [1, 8, 10, 14]
        var tree = fromArray(1, 3, 2, 4);
        tree.update(1, 7f);

        for (int i = 0; i < 50; i++) {
            assertEquals(1, tree.sampleIndexByValueInRange(1.0f + DELTA, 8.0f));
            assertEquals(2, tree.sampleIndexByValueInRange(8.0f + DELTA, 10.0f));
        }
    }

    @Test
    void testDominantPriorityAlwaysSameIndex() {
        // [0.1, 0.1, 99.8, 0.1] -> idx=2 occupies (0.2, 100.0]
        var tree = fromArray(0.1f, 0.1f, 99.8f, 0.1f);
        for (int i = 0; i < 50; i++)
            assertEquals(2, tree.sampleIndexByValueInRange(0.2f + DELTA, 100.0f));
    }

    @Test
    void testSingleElementAlwaysIdx0() {
        var tree = create(1, 42f);
        for (int i = 0; i < 50; i++)
            assertEquals(0, tree.sampleIndexByValueInRange(DELTA, 42f));
    }

    @Test
    void testSampleIndexByValueInRangeAlwaysReturnsValidIndex() {
        var tree = fromArray(1, 2, 3, 4, 5);
        int n = tree.size();
        float total = tree.sum();

        for (int i = 0; i < 1000; i++) {
            int idx = tree.sampleIndexByValueInRange(DELTA, total);
            assertTrue(idx >= 0 && idx < n,
                    "idx=" + idx + " outside interval [0," + n + ")");
        }
    }

    @Test
    void testSampleIndexByValueInRangeDistributionBias() {
        // idx=4 has priority 80/85 -> it must dominate all samples
        var tree = fromArray(1, 1, 1, 1, 80);
        float total = tree.sum();

        int countLast = 0;
        int trials = 5000;
        for (int i = 0; i < trials; i++)
            if (tree.sampleIndexByValueInRange(DELTA, total) == 4) countLast++;

        assertTrue(countLast > trials * 0.85,
                "idx=4 appeared only " + countLast + "/" + trials + " times");
    }

    @Test
    void testSampleAfterZeroingPriorities() {
        var tree = fromArray(5, 5, 5, 5, 1);
        tree.update(0, 0f);
        tree.update(1, 0f);
        tree.update(2, 0f);
        tree.update(3, 0f);

        float total = tree.sum();
        for (int i = 0; i < 200; i++)
            assertEquals(4, tree.sampleIndexByValueInRange(DELTA, total));
    }

    @Test
    void testInitValuePopulatesAllSlots() {
        // all slots start with initValue=3 -> uniform prefix sums [3, 6, 9, 12]
        var tree = create(4, 3f);
        for (int i = 0; i < 50; i++) {
            assertEquals(0, tree.sampleIndexByValueInRange(DELTA,        3.0f));
            assertEquals(1, tree.sampleIndexByValueInRange(3.0f + DELTA, 6.0f));
            assertEquals(2, tree.sampleIndexByValueInRange(6.0f + DELTA, 9.0f));
            assertEquals(3, tree.sampleIndexByValueInRange(9.0f + DELTA, 12.0f));
        }
    }

    // =========================================================
    // Miscellaneous
    // =========================================================

    @Test
    void testFloatPriorities() {
        var tree = fromArray(0.5f, 1.5f, 0.25f, 2.75f);
        assertEquals(5.0f,  tree.sum(),          DELTA);
        assertEquals(2.0f,  tree.rangeSum(0, 1), DELTA);
        assertEquals(2.25f, tree.prefixSum(2),   DELTA);
    }

    @Test
    void testLargeArray() {
        int n = 100;
        var tree = create(n, 0f);
        float expected = 0;
        for (int i = 0; i < n; i++) {
            tree.update(i, i + 1f);
            expected += i + 1f;
        }
        assertEquals(expected, tree.rangeSum(0, 99), DELTA);

        tree.update(50, 1000f);
        assertEquals(expected - 51f + 1000f, tree.rangeSum(0, 99), DELTA);
    }

    @Test
    void testOddSizeArray() {
        var tree = fromArray(10, 20, 30, 40, 50);
        assertEquals(60f,  tree.rangeSum(0, 2), DELTA);
        assertEquals(90f,  tree.rangeSum(1, 3), DELTA);
        assertEquals(120f, tree.rangeSum(2, 4), DELTA);
        assertEquals(150f, tree.rangeSum(0, 4), DELTA);
    }

    @Test
    void testArrayWithZeros() {
        var tree = fromArray(0, 5, 0, 3, 0, 7);
        assertEquals(0f,  tree.rangeSum(0, 0), DELTA);
        assertEquals(5f,  tree.rangeSum(1, 1), DELTA);
        assertEquals(8f,  tree.rangeSum(1, 3), DELTA);
        assertEquals(15f, tree.rangeSum(0, 5), DELTA);
    }

    @Test
    void testRandomizedRangeSum() {
        var rng = new Random(5);

        for (int time = 0; time < 500; time++) {
            int n = 1 + rng.nextInt(200);
            float[] array = new float[n];
            var tree = create(n, 0f);

            for (int i = 0; i < n; i++) {
                array[i] = rng.nextInt(100);
                tree.update(i, array[i]);
            }

            for (int q = 0; q < 50; q++) {
                if (rng.nextBoolean()) {
                    int a = rng.nextInt(n);
                    int b = rng.nextInt(n);
                    int start = Math.min(a, b);
                    int end = Math.max(a, b);

                    float exp = 0;
                    for (int i = start; i <= end; i++) exp += array[i];
                    assertEquals(exp, tree.rangeSum(start, end), DELTA);
                } else {
                    int idx = rng.nextInt(n);
                    float val = rng.nextInt(100);
                    array[idx] = val;
                    tree.update(idx, val);
                }
            }
        }
    }
}
