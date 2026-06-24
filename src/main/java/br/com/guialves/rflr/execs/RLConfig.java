package br.com.guialves.rflr.execs;

public record RLConfig(
        String envName,
        int observations,
        int actions,
        float learningRate,
        float epsilon,
        float minEpsilon,
        float epsilonDecay,
        float gamma,
        int updateQTargetAtTimeN,
        int batchSize,
        int framesLimit,
        int bufferCapacity,
        boolean saveModel,
        String algorithmName
) {

}
