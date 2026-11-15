package br.com.guialves.rflr.python;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static br.com.guialves.rflr.python.PythonDataStructures.toBoolArray;
import static br.com.guialves.rflr.python.PythonDataStructures.toLongArray;
import static br.com.guialves.rflr.python.PythonDataStructures.toMap;
import static br.com.guialves.rflr.python.PythonRuntime.callFunction;
import static br.com.guialves.rflr.python.PythonRuntime.decRef;
import static br.com.guialves.rflr.python.PythonRuntime.eval;
import static br.com.guialves.rflr.python.PythonRuntime.exec;
import static br.com.guialves.rflr.python.PythonRuntime.initPython;
import static br.com.guialves.rflr.python.PythonRuntime.insideGil;
import static br.com.guialves.rflr.python.PythonRuntime.toLong;
import static org.bytedeco.cpython.global.python.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * NumPy support of {@link PythonDataStructures#toMap(PyObject)}.
 *
 * <h2>Why this matters</h2>
 * <p>
 * The vectorized environments of Gymnasium return an {@code info} dictionary whose values
 * are NumPy masks, e.g. {@code {"lives": array([3, 2, 0]), "_lives": array([True, True,
 * False])}}. The generic sequence branch used to convert them with
 * {@code PySequence_GetItem()} (which returns NumPy <i>scalars</i>) followed by
 * {@code PyLong_AsLong()}.
 * </p>
 *
 * <p>
 * {@code PyLong_AsLong(numpy.bool)} raises
 * {@code TypeError: 'numpy.bool' object cannot be interpreted as an integer} — NumPy 2.x
 * removed {@code __index__} from {@code numpy.bool} — and the error was never checked, so
 * it stayed <b>pending</b> in the CPython error indicator. CPython 3.12 converts any later
 * call that returns a value into {@code SystemError: ... returned a result with an
 * exception set}, which is exactly the failure reported by
 * {@code Env.renderVec()}: the {@code render()} call died because of an error raised
 * earlier, while converting the {@code info} of the previous step.
 * </p>
 *
 * <p>
 * Measured against CPython 3.12 / NumPy 2.3.5 (the versions embedded in this project).
 * </p>
 */
@DisplayName("PythonDataStructures: toMap with NumPy objects")
class PythonDataStructuresNumpyTest {

    @BeforeAll
    static void setUp() {
        initPython();
        assertDoesNotThrow(() -> insideGil(() -> exec("""
                import numpy as np
                """)));
    }

    @Test
    @DisplayName("toMap converts a NumPy bool array into a boolean[]")
    void toMapShouldConvertNumpyBoolArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'_mask': np.array([True, False, True, False])}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                assertEquals(1, map.size());
                assertArrayEquals(new boolean[]{true, false, true, false},
                        (boolean[]) map.get("_mask"),
                        "the NumPy bool mask must be converted to the real booleans");
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toMap converts a NumPy int64 array into a long[]")
    void toMapShouldConvertNumpyIntArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'lives': np.array([3, 2, 0], dtype=np.int64)}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                assertArrayEquals(new long[]{3L, 2L, 0L},
                        (long[]) map.get("lives"));
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toMap converts a NumPy float64 array into a double[]")
    void toMapShouldConvertNumpyFloatArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'scores': np.array([1.5, -2.25], dtype=np.float64)}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                assertArrayEquals(new double[]{1.5, -2.25},
                        (double[]) map.get("scores"), 0.0001);
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toMap converts a NumPy scalar into a single Java value")
    void toMapShouldConvertNumpyScalars() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'flag': np.bool_(True), 'counter': np.int64(7), 'ratio': np.float64(0.5)}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                assertEquals(Boolean.TRUE, map.get("flag"));
                assertEquals(7L, map.get("counter"));
                assertEquals(0.5, (Double) map.get("ratio"), 0.0001);
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toMap converts the info of a vectorized environment")
    void toMapShouldConvertVectorizedEnvInfo() {
        assertDoesNotThrow(() -> insideGil(() -> {
            // The exact shape returned by gymnasium's SyncVectorEnv.step().
            exec("""
            info = {
                'lives': np.array([3, 2, 0, 1], dtype=np.int64),
                '_lives': np.array([True, True, False, True]),
                'frame_number': np.array([10, 20, 30, 40], dtype=np.int64),
                '_frame_number': np.array([True, False, True, False]),
            }
            """);
            var info = eval("info");
            try {
                var map = toMap(info);
                assertEquals(4, map.size());
                assertArrayEquals(new long[]{3L, 2L, 0L, 1L}, (long[]) map.get("lives"));
                assertArrayEquals(new boolean[]{true, true, false, true},
                        (boolean[]) map.get("_lives"));
                assertArrayEquals(new long[]{10L, 20L, 30L, 40L},
                        (long[]) map.get("frame_number"));
                assertArrayEquals(new boolean[]{true, false, true, false},
                        (boolean[]) map.get("_frame_number"));
            } finally {
                decRef(info);
            }
        }));
    }

    @Test
    @DisplayName("toMap converts a NumPy array nested in a dictionary")
    void toMapShouldConvertNestedNumpyArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'outer': {'_mask': np.array([True, False])}}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                var outer = (java.util.Map<?, ?>) map.get("outer");
                assertArrayEquals(new boolean[]{true, false},
                        (boolean[]) outer.get("_mask"));
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toMap keeps converting a Python list of integers into a long[]")
    void toMapShouldKeepConvertingPythonLists() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'a': 1, 'b': [2, 3, 4], 'c': 'text'}");
            var dict = eval("d");
            try {
                var map = toMap(dict);
                assertEquals(1L, map.get("a"));
                assertArrayEquals(new long[]{2L, 3L, 4L}, (long[]) map.get("b"));
                assertEquals("text", map.get("c"));
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("toBoolArray converts a NumPy bool array")
    void toBoolArrayShouldConvertNumpyBoolArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("arr = np.array([True, False, False, True])");
            var arr = eval("arr");
            try {
                assertArrayEquals(new boolean[]{true, false, false, true},
                        toBoolArray(arr));
            } finally {
                decRef(arr);
            }
        }));
    }

    @Test
    @DisplayName("toLongArray converts a NumPy int64 array")
    void toLongArrayShouldConvertNumpyIntArray() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("arr = np.array([10, 20, 30], dtype=np.int64)");
            var arr = eval("arr");
            try {
                assertArrayEquals(new long[]{10L, 20L, 30L}, toLongArray(arr));
            } finally {
                decRef(arr);
            }
        }));
    }

    @Test
    @DisplayName("toMap with a NumPy bool array leaves no pending exception")
    void toMapShouldNotLeavePendingException() {
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("d = {'_mask': np.array([True, False])}");
            var dict = eval("d");
            try {
                toMap(dict);
                assertNull(PyErr_Occurred(),
                        "the error indicator must be clean: a pending exception would "
                                + "break the next call that returns a value");
            } finally {
                decRef(dict);
            }
        }));
    }

    @Test
    @DisplayName("a call that returns a value still works after toMap with a NumPy bool array")
    void callThatReturnsAValueShouldStillWorkAfterToMap() {
        assertDoesNotThrow(() -> insideGil(() -> {
            // This is the Env.renderVec() scenario: the info of the previous step is
            // converted by toMap() and the render() call must still work.
            exec("""
            info = {'_lives': np.array([True, False, True, False])}
            def render_frame():
                return 42
            """);
            var info = eval("info");
            var render = eval("render_frame");
            try {
                toMap(info);

                var frame = callFunction(render);
                try {
                    assertEquals(42L, toLong(frame),
                            "the render() call must not be poisoned by a pending exception");
                } finally {
                    decRef(frame);
                }
            } finally {
                decRef(info);
                decRef(render);
            }
        }));
    }

    @Test
    @DisplayName("PyLong_AsLong on a NumPy bool fails and leaves the exception pending")
    void pyLongAsLongOnNumpyBoolLeavesTheExceptionPending() {
        // Documents the trap that motivated the fix: PyLong_AsLong() returns -1 and
        // leaves the TypeError pending, which breaks the next call that returns a value.
        assertDoesNotThrow(() -> insideGil(() -> {
            exec("flag = np.bool_(True)");
            var flag = eval("flag");
            try {
                assertEquals(-1L, PyLong_AsLong(flag));
                assertNotNull(PyErr_Occurred(),
                        "the TypeError must be pending after the failed conversion");
            } finally {
                decRef(flag);
                // Drain it: a pending exception would poison the other tests.
                PyErr_Clear();
            }
        }));
    }
}
