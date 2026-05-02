package br.com.guialves.rflr.datastructure;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Implementation based on the tutorial "AlgorithmsThread 3: Segment Trees" and "Rainbow in All You Need":
 *  <a href="https://www.youtube.com/watch?v=QvgpIX4_vyA&t=70s">...</a>
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py">...</a>
 */
public class SumSegmentTree {

    private final int size;
    private final Node root;
    private final ThreadLocalRandom random;

    public SumSegmentTree(int start, int end, float[] array) {
        if (end >= array.length) throw new IllegalArgumentException("end must be less than array.length");
        this.size = array.length;
        this.root = new Node(start, end, array);
        this.random = ThreadLocalRandom.current();
    }

    public SumSegmentTree(float[] array) {
        this.size = array.length;
        this.root = new Node(0, array.length - 1, array);
        this.random = ThreadLocalRandom.current();
    }

    public int size() {
        return size;
    }

    public void update(int idx, float value) {
        root.update(idx, value);
    }

    public float sum() {
        return root.sum;
    }

    public float rangeSum(int start, int end) {
        return root.rangeSum(start, end);
    }

    public float prefixSum(int end) {
        return root.rangeSum(0, end);
    }

    public int sampleIndexByValueInRange(float lower, float upper) {
        float totalSum = sum();
        if (lower < 0) throw new IllegalArgumentException("lower cannot be negative: " + lower);
        if (upper > totalSum) throw new IllegalArgumentException("upper cannot exceed total sum. upper: " + upper + ", total: " + totalSum);
        if (lower >= upper) throw new IllegalArgumentException("lower must be < upper. lower: " + lower + ", upper: " + upper);
        float sample = random.nextFloat(lower, upper);
        return root.sampleIndexByValue(sample);
    }

    private static class Node {
        float sum;
        int leftMost, rightMost;
        Node left, right;

        Node(int start, int end, float[] array) {
            this.leftMost = start;
            this.rightMost = end;
            if (leftMost == rightMost) {
                this.sum = array[leftMost];
                return;
            }
            int mid = (start + end) / 2;
            this.left = new Node(start, mid, array);
            this.right = new Node(mid + 1, end, array);
            this.sum = left.sum + right.sum;
        }

        void update(int idx, float value) {
            if (idx < leftMost || idx > rightMost) return;
            if (leftMost == rightMost) {
                this.sum = value;
                return;
            }

            int mid = (leftMost + rightMost) / 2;
            if (idx <= mid) {
                left.update(idx, value);
            } else {
                right.update(idx, value);
            }

            this.sum = left.sum + right.sum;
        }

        float rangeSum(int start, int end) {
            if (start > rightMost || end < leftMost) return 0;
            if (start <= leftMost && end >= rightMost) return sum;
            return left.rangeSum(start, end) + right.rangeSum(start, end);
        }

        int sampleIndexByValue(float sample) {
            if (leftMost == rightMost) return leftMost;

            if (sample <= left.sum) {
                return left.sampleIndexByValue(sample);
            }

            return right.sampleIndexByValue(sample - left.sum);
        }
    }
}
