package br.com.guialves.rflr.python.numpy;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.numpy.NumPyByteBuffer.fillFromNumpy;
import static br.com.guialves.rflr.python.numpy.NumPyByteBuffer.onHeapBufferNumpy;
import static org.junit.jupiter.api.Assertions.*;

class NumPyByteBufferTest {

    @BeforeAll
    static void setUp() {
        initPython();
        insideGil(() -> assertDoesNotThrow(() -> exec("x = 1")));
    }

    @Test
    void testByteOrderCaching() {
        var buf1 = NumPyByteBuffer.onHeapBufferNumpy(100);
        var buf2 = NumPyByteBuffer.onHeapBufferNumpy(100);
        assertEquals(buf1.order(), buf2.order());
    }

    @Test
    void testByteOrderCachingDirect() {
        var buf1 = NumPyByteBuffer.offHeapBufferNumpy(100);
        var buf2 = NumPyByteBuffer.offHeapBufferNumpy(100);
        assertEquals(buf1.order(), buf2.order());
    }

    @Test
    void testSpecificArrayByteOrder() {
        exec("""
        import numpy as np
        arr_little = np.array([1, 2, 3], dtype='<f4')  # explicit Little-endian
        arr_big = np.array([1, 2, 3], dtype='>f4')     # explicit Big-endian
        """);

        try (var arrLittle = eval("arr_little");
             var arrBig = eval("arr_big")) {

            var bufLittle = onHeapBufferNumpy(arrLittle, 12);
            var bufBig = onHeapBufferNumpy(arrBig, 12);

            assertEquals(ByteOrder.LITTLE_ENDIAN, bufLittle.order());
            assertEquals(ByteOrder.BIG_ENDIAN, bufBig.order());
        }
    }

    @Test
    void testFillFromNumpy() {
        exec("""
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        """);

        try (var arr = eval("arr")) {
            var buffer = NumPyByteBuffer.onHeapBufferNumpy(5 * 4); // 5 floats * 4 bytes
            assertDoesNotThrow(() -> fillFromNumpy(arr, buffer));
            assertEquals(1.0f, buffer.getFloat(), 0.001);
            assertEquals(2.0f, buffer.getFloat(), 0.001);
            assertEquals(3.0f, buffer.getFloat(), 0.001);
            assertEquals(4.0f, buffer.getFloat(), 0.001);
            assertEquals(5.0f, buffer.getFloat(), 0.001);
        }
    }

    @Test
    void testFillFromNumpyWithInsufficientBuffer() {
        exec("""
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        """);

        try (var arr = eval("arr")) {
            ByteBuffer buffer = ByteBuffer.allocate(10);

            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> fillFromNumpy(arr, buffer)
            );

            assertTrue(exception.getMessage().contains("Buffer too small"));
        }
    }

    @Test
    void testFillFromNumpyNonContiguous() {
        exec("""
        import numpy as np
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        slice_arr = arr[:, 0]  # Non-contiguous slice
        """);

        try (var arr = eval("slice_arr")) {
            var buffer = NumPyByteBuffer.onHeapBufferNumpy(100);
            assertThrows(IllegalStateException.class, () -> fillFromNumpy(arr, buffer));
        }
    }

    @Test
    void testNumPyOperations() {
        exec("""
        import numpy as np
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        """);

        try (var c = eval("c")) {
            var buffer = NumPyByteBuffer.onHeapBufferNumpy(3 * 8); // 3 ints * 8 bytes (int64)
            fillFromNumpy(c, buffer);

            // NumPy default dtype é int64 (8 bytes)
            assertEquals(5L, buffer.getLong(0));
            assertEquals(7L, buffer.getLong(8));
            assertEquals(9L, buffer.getLong(16));
        }
    }

    @Test
    void testLargeArray() {
        exec("""
        import numpy as np
        large_arr = np.zeros(10000, dtype=np.float32)
        large_arr[0] = 1.0
        large_arr[9999] = 2.0
        """);

        try (var arr = eval("large_arr")) {
            var buffer = NumPyByteBuffer.onHeapBufferNumpy(10000 * 4);
            fillFromNumpy(arr, buffer);

            assertEquals(1.0f, buffer.getFloat(0), 0.001);
            assertEquals(2.0f, buffer.getFloat(9999 * 4), 0.001);
        }
    }

    @Test
    void testToIntArray() {
        exec("""
        import numpy as np
        int_arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        """);

        try (var arr = eval("int_arr")) {
            int[] result = NumPyByteBuffer.toIntArray(arr);
            assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
        }
    }

    @Test
    void testToIntArrayWrongDtype() {
        exec("""
        import numpy as np
        int_arr = np.array([1, 2, 3], dtype=np.int64)
        """);

        try (var arr = eval("int_arr")) {
            var exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> NumPyByteBuffer.toIntArray(arr)
            );
            assertTrue(exception.getMessage().contains("Expected numpy int32 array"));
        }
    }

    @Test
    void testToLongArray() {
        exec("""
        import numpy as np
        long_arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        """);

        try (var arr = eval("long_arr")) {
            long[] result = NumPyByteBuffer.toLongArray(arr);
            assertArrayEquals(new long[]{10L, 20L, 30L, 40L, 50L}, result);
        }
    }

    @Test
    void testToLongArrayWrongDtype() {
        exec("""
        import numpy as np
        long_arr = np.array([1, 2, 3], dtype=np.int32)
        """);

        try (var arr = eval("long_arr")) {
            var exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> NumPyByteBuffer.toLongArray(arr)
            );
            assertTrue(exception.getMessage().contains("Expected numpy int64 array"));
        }
    }

    @Test
    void testToIntArrayNegativeValues() {
        exec("""
        import numpy as np
        int_arr = np.array([-1, -2, 0, 2, 1], dtype=np.int32)
        """);

        try (var arr = eval("int_arr")) {
            int[] result = NumPyByteBuffer.toIntArray(arr);
            assertArrayEquals(new int[]{-1, -2, 0, 2, 1}, result);
        }
    }

    @Test
    void testToLongArrayNegativeValues() {
        exec("""
        import numpy as np
        long_arr = np.array([-1, -2, 0, 2, 1], dtype=np.int64)
        """);

        try (var arr = eval("long_arr")) {
            long[] result = NumPyByteBuffer.toLongArray(arr);
            assertArrayEquals(new long[]{-1L, -2L, 0L, 2L, 1L}, result);
        }
    }
}
