package br.com.guialves.rflr.execs;

import lombok.Builder;

@Builder(toBuilder = true)
public record RLConfig(
        String envName,
        int observations,
        int actions,
        float learningRate,
        float maxEpsilon,
        float minEpsilon,
        float epsilonDecay,
        float discountFactor, // or gamma
        int updateQTargetAtTimeN,
        int batchSize,
        int framesLimit,
        int bufferCapacity,
        boolean saveModel,
        String algorithmName
) {

    public RLConfig build() {
        var epsilonLinearStep = (maxEpsilon - minEpsilon) / framesLimit;
        return new RLConfig(
                envName,
                observations,
                actions,
                learningRate,
                maxEpsilon,
                minEpsilon,
                epsilonLinearStep,
                discountFactor,
                updateQTargetAtTimeN,
                batchSize,
                framesLimit,
                bufferCapacity,
                saveModel,
                algorithmName
        );
    }
}
