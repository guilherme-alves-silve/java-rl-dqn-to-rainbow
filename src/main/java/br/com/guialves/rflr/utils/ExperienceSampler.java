package br.com.guialves.rflr.utils;

import java.util.HashSet;
import java.util.concurrent.ThreadLocalRandom;

public class ExperienceSampler {

    private final int startTries;

    public ExperienceSampler() {
        this(5);
    }

    /**
     * Constructor of ExperienceSampler, used to
     * sample elements from ExperienceReplayBuffer.
     * @param startTries When doing sampling without replacement, the number of tries that
     *                   the algorithm will retry sampling without repetition
     */
    public ExperienceSampler(int startTries) {
        this.startTries = startTries;
    }

    public Experience[] sample(Experience[] buffer, int sampleSize, boolean replacement) {
        return sample(buffer, sampleSize, 0, buffer.length, replacement);
    }

    public Experience[] sample(Experience[] buffer, int sampleSize, int endExclusive, boolean replacement) {
        if (replacement) return randomSampleReplacement(buffer, sampleSize, 0, endExclusive);
        return randomSampleWithoutReplacement(buffer, sampleSize, 0, endExclusive);
    }

    public Experience[] sample(Experience[] buffer, int sampleSize, int startInclusive, int endExclusive, boolean replacement) {
        if (replacement) return randomSampleReplacement(buffer, sampleSize, startInclusive, endExclusive);
        return randomSampleWithoutReplacement(buffer, sampleSize, startInclusive, endExclusive);
    }

    private Experience[] randomSampleReplacement(Experience[] buffer, int sampleSize, int startInclusive, int endExclusive) {
        var batch = new Experience[sampleSize];
        for (int i = 0; i < sampleSize; ++i) {
            batch[i] = buffer[ThreadLocalRandom.current().nextInt(startInclusive, endExclusive)];
        }

        return batch;
    }

    /**
     * Return samples of the buffer array.
     * Observation: Fisher–Yates shuffle could be faster, but usually the buffer has thousands
     * or millions of elements, the problem is that Fisher–Yates shuffle clones the
     * input array.
     */
    private Experience[] randomSampleWithoutReplacement(Experience[] buffer, int sampleSize, int startInclusive, int endExclusive) {
        if (buffer.length < sampleSize) throw new IllegalArgumentException("Sample size (" + sampleSize +
                ") cannot exceed buffer size (" + buffer.length + ") for sampling without replacement");
        var batch = new Experience[sampleSize];

        var selected = new HashSet<Integer>();

        int tries;
        int randomIdx;
        for (int i = 0; i < sampleSize; ++i) {
            tries = startTries;
            do {
                randomIdx = ThreadLocalRandom.current().nextInt(startInclusive, endExclusive);
                --tries;
            } while (!selected.add(randomIdx) && tries > 0);

            batch[i] = buffer[randomIdx];
        }

        return batch;
    }
}
