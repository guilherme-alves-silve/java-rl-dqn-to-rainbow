package br.com.guialves.rflr.datastructure;

/**
 * @author Guilherme Alves Silveira
 */
public interface ISumSegmentTree {
    int size();

    float sum();

    void update(int idx, float value);

    float rangeSum(int start, int end);

    float prefixSum(int end);

    int sampleIndexByValueInRange(float lower, float upper);
}
