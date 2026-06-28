package br.com.guialves.rflr.algorithms.buffer;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ExperienceSamplerTest {

    @Test
    void shouldSampleWithoutReplacement() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 5))
                    .thenReturn(2)  // 1ª chamada
                    .thenReturn(4)  // 2ª chamada
                    .thenReturn(0); // 3ª chamada

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            var result = sampler.sample(experiences, 3, true);

            assertEquals(3, result.length);

            assertSame(experiences[2], result[0]);
            assertSame(experiences[4], result[1]);
            assertSame(experiences[0], result[2]);

            var unique = new HashSet<Experience>();
            for (Experience exp : result) {
                assertTrue(unique.add(exp), "Encontrou elemento duplicado no batch");
            }

            verify(mockRandom, times(3)).nextInt(0, 5);
        }
    }

    @Test
    void shouldSampleWithReplacement() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 5))
                    .thenReturn(2)
                    .thenReturn(2)
                    .thenReturn(4);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            Experience[] result = sampler.sample(experiences, 3, true);

            assertEquals(3, result.length);

            assertSame(experiences[2], result[0]);
            assertSame(experiences[2], result[1]);
            assertSame(experiences[4], result[2]);

            verify(mockRandom, times(3)).nextInt(0, 5);
        }
    }

    @Test
    void shouldThrowExceptionWhenBatchSizeExceedsBufferSize() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> sampler.sample(experiences, 5, false)
            );

            assertEquals("Sample size (5) cannot exceed buffer size (3) for sampling without replacement",
                    exception.getMessage());

            verify(mockRandom, never()).nextInt(anyInt(), anyInt());
        }
    }

    @Test
    void shouldAllowBatchSizeGreaterThanBufferWhenWithReplacement() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 3))
                    .thenReturn(0)
                    .thenReturn(1)
                    .thenReturn(2)
                    .thenReturn(0)
                    .thenReturn(1);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            var result = sampler.sample(experiences, 5, true);

            assertEquals(5, result.length);
            assertSame(experiences[0], result[0]);
            assertSame(experiences[1], result[1]);
            assertSame(experiences[2], result[2]);
            assertSame(experiences[0], result[3]);
            assertSame(experiences[1], result[4]);

            verify(mockRandom, times(5)).nextInt(0, 3);
        }
    }

    @Test
    void shouldHandleBatchSizeZero() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
            };

            Experience[] result = sampler.sample(experiences, 0, true);

            assertEquals(0, result.length);
            verify(mockRandom, never()).nextInt(anyInt(), anyInt());
        }
    }

    @Test
    void shouldHandleBatchSizeOne() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 5)).thenReturn(3);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            Experience[] result = sampler.sample(experiences, 1, true);

            assertEquals(1, result.length);
            assertSame(experiences[3], result[0]);
            verify(mockRandom, times(1)).nextInt(0, 5);
        }
    }

    @Test
    void shouldNotModifyOriginalBuffer() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 5))
                    .thenReturn(2)
                    .thenReturn(4)
                    .thenReturn(0);

            var sampler = new ExperienceSampler();
            var originalExperiences = new Experience[]{
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
                    mock(Experience.class),
            };

            var originalCopy = originalExperiences.clone();

            sampler.sample(originalExperiences, 3, true);
            assertArrayEquals(originalCopy, originalExperiences);
        }
    }

    @Test
    void shouldAvoidInfinityLoopWithTriesInnerVar() {
        try (var mockRandomFactory = mockStatic(ThreadLocalRandom.class)) {
            var mockRandom = mock(ThreadLocalRandom.class);
            mockRandomFactory.when(ThreadLocalRandom::current).thenReturn(mockRandom);

            when(mockRandom.nextInt(0, 5))
                    .thenReturn(2, 2, 2, 1, 1, 1, 0, 0, 0);

            var sampler = new ExperienceSampler();
            var experiences = new Experience[]{
                    mock(Experience.class, "exp0"),
                    mock(Experience.class, "exp1"),
                    mock(Experience.class, "exp2"),
                    mock(Experience.class, "exp3"),
                    mock(Experience.class, "exp4"),
            };

            sampler.sample(experiences, 3, false);

            verify(mockRandom, times(7)).nextInt(0, 5);
        }
    }

    @Test
    void shouldNeverHaveDuplicates() {
        var sampler = new ExperienceSampler();
        var experiences = createMockBuffer(100);

        for (int attempt = 0; attempt < 1000; attempt++) {
            var batch = sampler.sample(experiences, 10, false);

            var unique = new HashSet<>(Arrays.asList(batch));
            assertEquals(batch.length, unique.size(),
                    "Found duplicate in attempt " + attempt);
        }
    }

    private Experience[] createMockBuffer(int size) {
        var buffer = new Experience[size];
        for (int i = 0; i < size; i++) {
            buffer[i] = mock(Experience.class);
        }
        return buffer;
    }
}
