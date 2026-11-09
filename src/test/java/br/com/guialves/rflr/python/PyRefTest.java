package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.bytedeco.cpython.global.python.*;
import static org.junit.jupiter.api.Assertions.*;

public class PyRefTest {

    private static final int ITERATIONS = 1000;

    /** The {@code __main__} globals, a borrowed reference. */
    private static PyObject globals;
    /** The {@code holder} instance, a borrowed reference. */
    private static PyObject holder;

    @BeforeAll
    public static void setUp() throws Exception {
        boolean initialized = org.bytedeco.cpython.helper.python.Py_Initialize(
                org.bytedeco.cpython.presets.python.cachePackages());
        assertTrue(initialized, "CPython failed to initialize");

        globals = PyModule_GetDict(PyImport_AddModule("__main__"));
        assertFalse(globals == null || globals.isNull(), "globals is null");

        var code = PyRun_String(
                "class Holder:\n    pass\nh = Holder()\nh.obj = object()\n",
                Py_file_input, globals, globals);
        assertFalse(code == null || code.isNull(), "the script failed to run");
        Py_DECREF(code);

        holder = PyDict_GetItemString(globals, "h");
        assertFalse(holder == null || holder.isNull(), "holder is null");
    }

    private static long objRefCount() {
        PyObject obj = PyObject_GetAttrString(holder, "obj");
        assertFalse(obj == null || obj.isNull(), "holder.obj is null");
        long count = Py_REFCNT(obj);
        Py_DECREF(obj);
        return count - 1;
    }

    @Test
    public void testCloseOnPlainPyObjectDoesNotRelease() {
        long before = objRefCount();
        for (int i = 0; i < ITERATIONS; i++) {
            try (PyObject ignored = PyObject_GetAttrString(holder, "obj")) {
                // close() does nothing: the reference is not released
            }
        }

        long expectedLeak = before + ITERATIONS;
        assertEquals(expectedLeak, objRefCount(), "close() on a plain PyObject leaked " + (objRefCount() - before)
                + " references");
    }

    @Test
    public void testPyRefReleasesTheReference() {
        long before = objRefCount();
        for (int i = 0; i < ITERATIONS; i++) {
            try (var ref = PyRef.of(PyObject_GetAttrString(holder, "obj"))) {
                assertEquals(before + 1, ref.refCount());
            }
        }

        assertEquals(before, objRefCount(), "PyRef leaked references");
    }

    @Test
    public void testPyRefCloseIsIdempotent() {
        long before = objRefCount();
        var ref = PyRef.of(PyObject_GetAttrString(holder, "obj"));
        assertEquals(before + 1, ref.refCount());

        ref.close();
        assertTrue(ref.isNull(), "the reference was not invalidated");
        assertEquals(-1L, ref.refCount());
        assertEquals(before, objRefCount(), "the reference was not released");

        ref.close();
        assertEquals(before, objRefCount(), "the reference was released twice");
    }
}
