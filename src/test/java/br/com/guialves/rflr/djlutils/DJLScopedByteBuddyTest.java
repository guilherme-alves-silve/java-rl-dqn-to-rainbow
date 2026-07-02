package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.djlutils.bytebuddyfixtures.ScopedLeakyComputation;
import br.com.guialves.rflr.djlutils.bytebuddyfixtures.ScopedReturningComputation;
import br.com.guialves.rflr.djlutils.bytebuddyfixtures.UnscopedLeakyComputation;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.managedArrayCount;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DJLScopedByteBuddyTest {

    @Test
    void shouldShowTheLeakWithoutDJLScopedInstrumentation() {
        try (var manager = NDManager.newBaseManager()) {
            var leaky = new UnscopedLeakyComputation();
            int before = managedArrayCount(manager);

            leaky.calculate(manager);

            assertTrue(managedArrayCount(manager) > before);
        }
    }

    @Test
    void shouldCloseTemporaryArraysWithInjectedScopedCleanup() {
        DJLScopedByteBuddy.installOn(ScopedLeakyComputation.class);

        try (var manager = NDManager.newBaseManager()) {
            var scoped = new ScopedLeakyComputation();
            int before = managedArrayCount(manager);

            scoped.calculate(manager);
            int afterStep1 = managedArrayCount(manager);
            scoped.calculate(manager);
            int afterStep2 = managedArrayCount(manager);

            assertEquals(before, afterStep1);
            assertEquals(afterStep1, afterStep2);
        }
    }

    @Test
    void shouldKeepReturnedArrayAliveAndCloseOnlyIntermediates() {
        DJLScopedByteBuddy.installOn(ScopedReturningComputation.class);

        try (var manager = NDManager.newBaseManager()) {
            var scoped = new ScopedReturningComputation();
            int before = managedArrayCount(manager);

            var result = scoped.calculate(manager);

            assertEquals(before + 1, managedArrayCount(manager));
            assertEquals(4f, result.getFloat(), 0.0001f);

            result.close();
            assertEquals(before, managedArrayCount(manager));
        }
    }
}
