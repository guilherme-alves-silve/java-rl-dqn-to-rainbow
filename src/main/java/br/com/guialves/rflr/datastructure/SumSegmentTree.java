package br.com.guialves.rflr.datastructure;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Implementation based on the tutorial "AlgorithmsThread 3: Segment Trees" and "Rainbow in All You Need":
 *  <a href="https://www.youtube.com/watch?v=QvgpIX4_vyA">...</a>
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py">...</a>
 * @author Guilherme Alves Silveira
 */
public class SumSegmentTree implements ISumSegmentTree {

    private final int size;
    private final Node root;

    public SumSegmentTree(int size) {
        this(size, 0.0f);
    }

    public SumSegmentTree(int size, float initValue) {
        this.size = size;
        this.root = new Node(0, size - 1, initValue);
    }

    @Override
    public int size() { return size; }

    @Override
    public float sum() { return root.sum; }

    @Override
    public void update(int idx, float value) {
        update(root, idx, value);
    }

    public float get(int idx) {
        if (idx < 0 || idx >= size) {
            throw new IndexOutOfBoundsException("Index: " + idx + ", Size: " + size);
        }
        return get(root, idx);
    }

    @Override
    public float rangeSum(int start, int end) {
        return rangeSum(root, start, end);
    }

    @Override
    public float prefixSum(int end) {
        return rangeSum(root, 0, end);
    }

    @Override
    public int sampleIndexByValueInRange(float lower, float upper) {
        float totalSum = sum();
        if (lower < 0) throw new IllegalArgumentException("lower cannot be negative: " + lower);
        if (upper > totalSum) throw new IllegalArgumentException("upper cannot exceed total sum. upper: " + upper + ", total: " + totalSum);
        if (lower >= upper) throw new IllegalArgumentException("lower must be < upper. lower: " + lower + ", upper: " + upper);
        float sample = ThreadLocalRandom.current().nextFloat(lower, upper);
        return sampleIndexByValue(root, sample);
    }

    private void update(Node node, int idx, float value) {
        if (idx < node.leftMost || idx > node.rightMost) return;
        if (node.leftMost == node.rightMost) {
            node.sum = value;
            return;
        }
        int mid = (node.leftMost + node.rightMost) / 2;
        if (idx <= mid) {
            update(node.left, idx, value);
        } else {
            update(node.right, idx, value);
        }
        node.sum = node.left.sum + node.right.sum;
    }

    private float get(Node node, int idx) {
        if (node.leftMost == node.rightMost) {
            return node.sum;
        }
        int mid = (node.leftMost + node.rightMost) / 2;
        if (idx <= mid) {
            return get(node.left, idx);
        }

        return get(node.right, idx);
    }

    private float rangeSum(Node node, int start, int end) {
        if (start > node.rightMost || end < node.leftMost) return 0;
        if (start <= node.leftMost && end >= node.rightMost) return node.sum;
        return rangeSum(node.left, start, end) + rangeSum(node.right, start, end);
    }

    private int sampleIndexByValue(Node node, float sample) {
        if (node.leftMost == node.rightMost) return node.leftMost;
        if (sample <= node.left.sum) return sampleIndexByValue(node.left, sample);
        return sampleIndexByValue(node.right, sample - node.left.sum);
    }

    private static class Node {
        float sum;
        int leftMost, rightMost;
        Node left, right;

        Node(int start, int end, float initValue) {
            this.leftMost = start;
            this.rightMost = end;
            if (leftMost == rightMost) {
                this.sum = initValue;
                return;
            }
            int mid = (start + end) / 2;
            this.left = new Node(start, mid, initValue);
            this.right = new Node(mid + 1, end, initValue);
            this.sum = left.sum + right.sum;
        }
    }
}
