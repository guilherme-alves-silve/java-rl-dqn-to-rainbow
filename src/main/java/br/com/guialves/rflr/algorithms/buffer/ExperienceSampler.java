package br.com.guialves.rflr.algorithms.buffer;

import java.util.HashSet;
import java.util.concurrent.ThreadLocalRandom;

public class ExperienceSampler {

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

        if (startInclusive < 0 || endExclusive > buffer.length || startInclusive > endExclusive) {
            throw new IllegalArgumentException("Invalid sample range");
        }
        int populationSize = endExclusive - startInclusive;
        if (sampleSize > populationSize) {
            throw new IllegalArgumentException("Sample size (" + sampleSize +
                    ") cannot exceed population size (" + populationSize +
                    ") for sampling without replacement");
        }

        var batch = new Experience[sampleSize];

        var selected = new HashSet<Integer>();

        int randomIdx;
        for (int i = 0; i < sampleSize; ++i) {
            do {
                randomIdx = ThreadLocalRandom.current().nextInt(startInclusive, endExclusive);
            } while (!selected.add(randomIdx));

            batch[i] = buffer[randomIdx];
        }

        return batch;
    }
}
