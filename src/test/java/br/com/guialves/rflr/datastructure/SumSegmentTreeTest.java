package br.com.guialves.rflr.datastructure;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

class SumSegmentTreeTest {

    private static final float DELTA = 1e-5f;

    private SumSegmentTree tree5;   // [1, 2, 3, 4, 5]
    private SumSegmentTree tree8;   // [1, 2, 3, 4, 5, 6, 7, 8]

    @BeforeEach
    void setUp() {
        tree5 = new SumSegmentTree(new float[]{1, 2, 3, 4, 5});
        tree8 = new SumSegmentTree(new float[]{1, 2, 3, 4, 5, 6, 7, 8});
    }

    @Test
    void testConstructorDefaultRange() {
        assertNotNull(tree5);
        assertEquals(5, tree5.size());
        assertEquals(15f, tree5.rangeSum(0, 4), DELTA);
    }

    @Test
    void testConstructorExplicitRange() {
        float[] array = {10, 20, 30, 40, 50};
        var tree = new SumSegmentTree(1, 3, array);
        // covers indices 1..3 → sum = 20+30+40 = 90
        assertEquals(90f, tree.rangeSum(1, 3), DELTA);
    }

    @Test
    void testConstructorExplicitRangeThrowsWhenEndOutOfBounds() {
        float[] array = {1, 2, 3};
        assertThrows(IllegalArgumentException.class,
                () -> new SumSegmentTree(0, 5, array));
    }

    @Test
    void testConstructorSingleElement() {
        var tree = new SumSegmentTree(new float[]{10f});
        assertEquals(1, tree.size());
        assertEquals(10f, tree.rangeSum(0, 0), DELTA);
    }

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

    @Test
    void testPrefixSumDefinitionConsistency() {
        float[] array = {3, 1, 4, 1, 5};
        var tree = new SumSegmentTree(array);

        float running = 0;
        for (int i = 0; i < array.length; i++) {
            running += array[i];
            assertEquals(running, tree.prefixSum(i), DELTA);
        }
    }

    @Test
    void testPrefixSumAfterMultipleUpdates() {
        float[] array = {1, 2, 3, 4, 5};
        var tree = new SumSegmentTree(array);

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
        // prefixSum(k) == np.cumsum(array)[k]
        float[] array = {5, 3, 8, 2, 7};
        var tree = new SumSegmentTree(array);

        float[] cumsum = {5, 8, 16, 18, 25};
        for (int i = 0; i < cumsum.length; i++) {
            assertEquals(cumsum[i], tree.prefixSum(i), DELTA);
        }
    }

    @Test
    void testSampleIndexByValueInRangeAlwaysReturnsValidIndex() {
        var tree = new SumSegmentTree(new float[]{1, 2, 3, 4, 5});
        int n = tree.size();
        float total = tree.sum();

        for (int i = 0; i < 1000; i++) {
            int idx = tree.sampleIndexByValueInRange(0f, total);
            assertTrue(idx >= 0 && idx < n,
                    "idx=" + idx + " fora do intervalo [0," + n + ")");
        }
    }

    @Test
    void testSampleIndexByValueInRangeDistributionBias() {
        // idx=4 tem prioridade 80 de 85 total → deve dominar amostras
        var tree = new SumSegmentTree(new float[]{1, 1, 1, 1, 80});
        float total = tree.sum();

        int countLast = 0;
        int trials = 5000;
        for (int i = 0; i < trials; i++) {
            if (tree.sampleIndexByValueInRange(0f, total) == 4) countLast++;
        }

        // Espera ~94% de hits no idx=4; aceita acima de 85% (margem conservadora)
        assertTrue(countLast > trials * 0.85,
                "idx=4 apareceu apenas " + countLast + "/" + trials + " vezes");
    }

    @Test
    void testSampleIndexByValueInRangeAfterUpdate() {
        // Após zerar prioridades de 0..3, apenas idx=4 deve ser amostrado
        var tree = new SumSegmentTree(new float[]{5, 5, 5, 5, 1});
        tree.update(0, 0f);
        tree.update(1, 0f);
        tree.update(2, 0f);
        tree.update(3, 0f);

        float total = tree.sum();
        for (int i = 0; i < 200; i++) {
            // qualquer sample em (0, total] deve cair em idx=4
            int idx = tree.sampleIndexByValueInRange(0f, total);
            assertEquals(4, idx);
        }
    }

    @Test
    void testLowerBoundInvariantRandomized() {
        // Para qualquer valor k em (0, total], o índice i retornado deve satisfazer:
        //   prefixSum(i)   >= k
        //   prefixSum(i-1) <  k  (se i > 0)
        var rng = new Random(42);

        for (int t = 0; t < 300; t++) {
            int n = 1 + rng.nextInt(200);
            float[] array = new float[n];
            for (int i = 0; i < n; i++) array[i] = rng.nextInt(10);

            var tree = new SumSegmentTree(array);
            float total = tree.sum();
            if (total == 0) continue;

            for (int q = 0; q < 50; q++) {
                float sample = rng.nextFloat() * total;
                int idx = tree.sampleIndexByValueInRange(0f, total);

                assertTrue(idx >= 0 && idx < n);
                assertTrue(tree.prefixSum(idx) >= sample);
            }
        }
    }

    @Test
    void testUniformArray() {
        // Array uniform array [2, 2, 2, 2] → prefix sums [2, 4, 6, 8]
        var tree = new SumSegmentTree(new float[]{2, 2, 2, 2});
        for (int i = 0; i < 50; i++) {
            assertEquals(0, tree.sampleIndexByValueInRange(DELTA, 2.0f));
            assertEquals(1, tree.sampleIndexByValueInRange(2.0f + DELTA, 4.0f));
            assertEquals(2, tree.sampleIndexByValueInRange(4.0f + DELTA, 6.0f));
            assertEquals(3, tree.sampleIndexByValueInRange(6.0f + DELTA, 8.0f));
        }
    }

    @Test
    void testLargeArray() {
        int n = 100;
        float[] array = new float[n];
        for (int i = 0; i < n; i++) array[i] = i + 1;

        var tree = new SumSegmentTree(array);
        assertEquals(5050f, tree.rangeSum(0, 99), DELTA);

        tree.update(50, 1000f);
        assertEquals(5050f - 51f + 1000f, tree.rangeSum(0, 99), DELTA);
    }

    @Test
    void testOddSizeArray() {
        var tree = new SumSegmentTree(new float[]{10, 20, 30, 40, 50});
        assertEquals(60f,  tree.rangeSum(0, 2), DELTA);
        assertEquals(90f,  tree.rangeSum(1, 3), DELTA);
        assertEquals(120f, tree.rangeSum(2, 4), DELTA);
        assertEquals(150f, tree.rangeSum(0, 4), DELTA);
    }

    @Test
    void testArrayWithZeros() {
        var tree = new SumSegmentTree(new float[]{0, 5, 0, 3, 0, 7});
        assertEquals(0f,  tree.rangeSum(0, 0), DELTA);
        assertEquals(5f,  tree.rangeSum(1, 1), DELTA);
        assertEquals(8f,  tree.rangeSum(1, 3), DELTA);
        assertEquals(15f, tree.rangeSum(0, 5), DELTA);
    }

    @Test
    void testFloatPriorities() {
        // PER usa floats fracionários; garante que somas fracionárias são preservadas
        var tree = new SumSegmentTree(new float[]{0.5f, 1.5f, 0.25f, 2.75f});
        assertEquals(5.0f, tree.sum(), DELTA);
        assertEquals(2.0f, tree.rangeSum(0, 1), DELTA);
        assertEquals(2.25f, tree.prefixSum(2), DELTA);
    }

    @Test
    void testRandomizedRangeSum() {
        var rng = new Random(5);

        for (int t = 0; t < 500; t++) {
            int n = 1 + rng.nextInt(200);
            float[] array = new float[n];
            for (int i = 0; i < n; i++) array[i] = rng.nextInt(100);

            var tree = new SumSegmentTree(array);

            for (int q = 0; q < 50; q++) {
                if (rng.nextBoolean()) {
                    int a = rng.nextInt(n), b = rng.nextInt(n);
                    int l = Math.min(a, b), r = Math.max(a, b);

                    float expected = 0;
                    for (int i = l; i <= r; i++) expected += array[i];
                    assertEquals(expected, tree.rangeSum(l, r), DELTA);
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
