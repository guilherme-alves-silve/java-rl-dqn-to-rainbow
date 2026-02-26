package br.com.guialves.rflr.python.numpy;

import org.junit.jupiter.api.*;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.numpy.NumPyByteBuffer.toDoubleArray;
import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class NumPyBufferViewTest {

    @BeforeAll
    static void setUp() {
        initPython();
        insideGil(() -> assertDoesNotThrow(() -> exec("x = 1")));
    }

    @Test
    @Order(1)
    void shouldCreateBufferViewFromNumPyArray() {
        exec("import numpy as np; arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)");
        try (var pyArr = eval("arr");
             var view = new NumPyBufferView(pyArr)) {
            assertTrue(view.capacity() > 0);
        }
    }

    @Test
    @Order(2)
    void shouldHaveCorrectBufferSize() {
        exec("import numpy as np; arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)");
        try (var pyArr = eval("arr");
             var view = new NumPyBufferView(pyArr)) {
            // float32 = 4 bytes, 3 elements = 12 bytes
            assertEquals(12, view.capacity());
        }
    }

    @Test
    @Order(3)
    void shouldReadCorrectValues() {
        exec("import numpy as np; arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)");
        try (var pyArr = eval("arr");
             var view = new NumPyBufferView(pyArr)) {
            var buffer = view.asFloatBuffer();
            assertEquals(1.0f, buffer.get(0), 0.0001f);
            assertEquals(2.0f, buffer.get(1), 0.0001f);
            assertEquals(3.0f, buffer.get(2), 0.0001f);
            buffer.put(0, 4f);
            buffer.put(1, 5f);
            buffer.put(2, 6f);
            assertArrayEquals(new double[]{4., 5., 6.}, toDoubleArray(pyArr), 0.0001);
        }
    }

    @Test
    @Order(4)
    void shouldWhenViewDeallocateTheArrShouldExists() {
        exec("import numpy as np; arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)");
        var pyArr = eval("arr");
        try (var view = new NumPyBufferView(pyArr)) {
            var buffer = view.asFloatBuffer();
            assertEquals(1.0f, buffer.get(0), 0.0001f);
            assertEquals(2.0f, buffer.get(1), 0.0001f);
            assertEquals(3.0f, buffer.get(2), 0.0001f);
        }
        assertArrayEquals(new double[]{1., 2., 3.}, toDoubleArray(pyArr), 0.0001);
        pyArr.close();
    }

    @Test
    @Order(5)
    void shouldThrowWhenInvalidPyObject() {
        try (var pyLong = eval("42")) {
            assertThrows(IllegalStateException.class, () -> new NumPyBufferView(pyLong));
        }
    }
}
