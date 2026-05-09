package br.com.guialves.rflr.datastructure;

/**
 * Implementation based on the tutorial "AlgorithmsThread 3: Segment Trees" and "Rainbow in All You Need":
 *  <a href="https://www.youtube.com/watch?v=QvgpIX4_vyA&t=70s">...</a>
 *  <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py">...</a>
 * @author Guilherme Alves Silveira
 */
public class MinSegmentTree {

    private final int size;
    private final Node root;

    public MinSegmentTree(int size) {
        this(size, Float.MAX_VALUE);
    }

    public MinSegmentTree(int size, float initValue) {
        this.size = size;
        this.root = new Node(0, size - 1, initValue);
    }

    public int size() { return size; }

    public float min() { return root.min; }

    public void update(int idx, float value) {
        update(root, idx, value);
    }

    private void update(Node node, int idx, float value) {
        if (idx < node.leftMost || idx > node.rightMost) return;
        if (node.leftMost == node.rightMost) {
            node.min = value;
            return;
        }
        int mid = (node.leftMost + node.rightMost) / 2;
        if (idx <= mid) {
            update(node.left, idx, value);
        } else {
            update(node.right, idx, value);
        }
        node.min = Math.min(node.left.min, node.right.min);
    }

    private static class Node {
        float min;
        int leftMost, rightMost;
        Node left, right;

        Node(int start, int end, float initValue) {
            this.leftMost = start;
            this.rightMost = end;
            if (leftMost == rightMost) {
                this.min = initValue;
                return;
            }
            int mid = (start + end) / 2;
            this.left = new Node(start, mid, initValue);
            this.right = new Node(mid + 1, end, initValue);
            this.min = Math.min(left.min, right.min);
        }
    }
}
