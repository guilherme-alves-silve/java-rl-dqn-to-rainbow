package br.com.guialves.rflr.python;

import io.vavr.CheckedFunction0;
import io.vavr.CheckedRunnable;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.javacpp.BytePointer;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static br.com.guialves.rflr.python.PythonTypeChecks.isPyNull;
import static org.bytedeco.cpython.global.python.*;

@Slf4j
public final class PythonRuntime {

    public static final String JAVA_RL_SITE_PACKAGES = "JAVA_RL_SITE_PACKAGES";
    private static final boolean DEBUG = false;
    private static boolean initialized = false;
    private static PyObject globals;

    private PythonRuntime() {
        throw new IllegalArgumentException("No PythonRuntime!");
    }

    private static File[] cachePackages() throws IOException {
        var path = org.bytedeco.cpython.presets.python.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);

        var sitePackages = System.getProperty(JAVA_RL_SITE_PACKAGES,
                System.getenv(JAVA_RL_SITE_PACKAGES));

        if (null == sitePackages) {
            throw new IllegalStateException("It's mandatory to pass property/env JAVA_RL_SITE_PACKAGES");
        }

        path[path.length - 1] = new File(sitePackages);
        return path;
    }

    @SneakyThrows
    public static synchronized void initPython() {
        if (initialized) return;
        if (Boolean.getBoolean("python.initialized")) {
            initialized = true;
            return;
        }

        initialized = Py_Initialize(cachePackages());
        globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        if (!initialized) {
            throw new IllegalArgumentException("PythonRuntime is not initialized!");
        }

        System.setProperty("python.initialized", "true");
    }

    /**
     * Py_Finalize() is intentionally not called due to known issues documented in the
     * Python C API documentation (<a href="https://docs.python.org/3/c-api/init.html">...</a>):
     * - "Dynamically loaded extension modules loaded by Python are not unloaded"
     * - "Some memory allocated by extension modules may not be freed"
     * - Random destruction order can cause crashes in extension module destructors
     * <p>
     * In practice, calling Py_Finalize() with extension modules like NumPy and Gymnasium
     * (which Gymnasium depends on) causes JVM crashes in our environment.
     * <p>
     * The OS will properly clean up all resources when the JVM process terminates.
     */
    public static synchronized void finalizePython() {
        if (!initialized) return;
        if (!Boolean.getBoolean("python.initialized")) {
            initialized = false;
            return;
        }

        try {
            if (globals != null) {
                globals.close();
                globals = null;
            }

            initialized = false;
            log.info("Python finalized successfully");
        } catch (Exception e) {
            log.error("Error during Python finalization", e);
        } finally {
            System.clearProperty("python.initialized");
        }
    }

    /**
     * For performance reasons, not all methods are
     * check inside the GIL (Global Interpreter Lock)
     */
    @SneakyThrows
    public static void insideGil(CheckedRunnable runnable) {
        var gstate = PyGILState_Ensure();
        try {
            runnable.run();
        } finally {
            PyGILState_Release(gstate);
        }
    }

    @SneakyThrows
    public static <R> R insideGil(CheckedFunction0<R> function0) {
        var gstate = PyGILState_Ensure();
        try {
            return function0.apply();
        } finally {
            PyGILState_Release(gstate);
        }
    }

    public static void exec(String code) {
        PyErr_Clear();
        try (var _ = PyRun_StringFlags(
                code,
                Py_file_input,
                globals,
                globals,
                null
        )) {
            checkError();
        }
    }

    public static void execIsolated(String code) {
        PyErr_Clear();
        var locals = PyDict_New();
        try (var _ = PyRun_StringFlags(code, Py_file_input, globals, locals, null)) {
            checkError();

            if (DEBUG) {
                printDict(locals, "local variable dict");
                printDict(globals, "global variable dict");
            }
        } finally {
            Py_DECREF(locals);
        }
    }

    public static PyObject eval(String expression) {
        PyErr_Clear();
        try (var result = PyRun_StringFlags(
                expression,
                Py_eval_input,
                globals,
                globals,
                null
        )) {
            checkError();
            return result;
        }
    }

    public static String toStr(PyObject obj) {
        return toStr(obj, "");
    }

    public static String toStr(PyObject obj, String msg) {
        if (null == obj || Py_IsNone(obj) == 1) {
            return null;
        }

        try (var pyStr = PyObject_Str(obj)) {
            if (pyStr == null) {
                PyErr_Print();
                throw new RuntimeException("PyObject_Str failed: " + msg);
            }

            try (var bytes = PyUnicode_AsUTF8String(pyStr)) {
                if (bytes == null) {
                    PyErr_Print();
                    throw new RuntimeException("PyUnicode_AsUTF8String failed: " + msg);
                }

                try (var temp = PyBytes_AsString(bytes)) {
                    return temp.getString();
                }
            }
        }
    }

    public static PyObject attr(PyObject obj, String attr) {
        var result = PyObject_GetAttrString(obj, attr);
        if (result == null || result.isNull()) {
            PyErr_Print();
            throw new IllegalArgumentException("Attribute not found: " + attr);
        }
        return result;
    }

    public static boolean hasAttr(PyObject obj, String name) {
        return PyObject_HasAttrString(obj, name) == 1;
    }

    public static String attrStr(PyObject obj, String attr) {
        try (var attrObj = PyObject_GetAttrString(obj, attr)) {
            return toStr(attrObj);
        }
    }

    public static long toLong(PyObject obj) {
        return PyLong_AsLong(obj);
    }

    public static double toDouble(PyObject obj) {
        return PyFloat_AsDouble(obj);
    }

    public static boolean toBool(PyObject obj) {
        return PyObject_IsTrue(obj) == 1;
    }

    public static PyObject callObj(PyObject obj) {
        if (obj == null || obj.isNull()) {
            throw new IllegalArgumentException("Cannot call null PyObject");
        }

        var result = PyObject_CallObject(obj, null);
        if (result == null || result.isNull()) {
            PyErr_Print();
            throw new RuntimeException("Failed to call PyObject");
        }
        return result.retainReference();
    }

    public static PyObject callMethod(PyObject obj, String method, PyObject... args) {
        if (obj == null || obj.isNull()) {
            throw new IllegalArgumentException("Cannot call method on null PyObject");
        }

        PyObject fn = null;
        try {
            fn = PyObject_GetAttrString(obj, method);
            if (fn == null || fn.isNull()) {
                PyErr_Print();
                throw new IllegalArgumentException("Method not found: " + method);
            }

            return callFunction(fn, args);
        } finally {
            if (fn != null && !fn.isNull()) {
                Py_DECREF(fn);
            }
        }
    }

    public static PyObject callMethodWithKwargs(PyObject obj, String method,
                                                PyObject[] args, PyObject kwargs) {
        if (obj == null || obj.isNull()) {
            throw new IllegalArgumentException("Cannot call method on null PyObject");
        }

        PyObject fn = null;
        try {
            fn = PyObject_GetAttrString(obj, method);
            if (fn == null || fn.isNull()) {
                PyErr_Print();
                throw new IllegalArgumentException("Method not found: " + method);
            }

            var tuple = args != null ? newArgs(args) : PyTuple_New(0);
            try {
                var result = PyObject_Call(fn, tuple, kwargs);
                if (result == null || result.isNull()) {
                    PyErr_Print();
                    throw new RuntimeException("Failed to call method: " + method);
                }
                return result.retainReference();
            } finally {
                Py_DECREF(tuple);
            }
        } finally {
            if (fn != null && !fn.isNull()) {
                Py_DECREF(fn);
            }
        }
    }

    public static PyObject callFunction(PyObject fn, PyObject... args) {
        if (isPyNull(fn)) {
            throw new IllegalArgumentException("Cannot call null function");
        }

        var tuple = newArgs(args);
        try {
            var result = PyObject_CallObject(fn, tuple);
            if (result == null || result.isNull()) {
                PyErr_Print();
                throw new RuntimeException("Failed to call function");
            }
            return result.retainReference();
        } finally {
            Py_DECREF(tuple);
        }
    }

    public static PyObject newArgs(PyObject... args) {
        var tuple = PyTuple_New(args.length);
        if (tuple == null || tuple.isNull()) {
            throw new RuntimeException("Failed to allocate tuple");
        }

        for (int i = 0; i < args.length; i++) {
            PyObject arg = args[i];
            if (isPyNull(arg)) {
                Py_DECREF(tuple);
                throw new IllegalArgumentException("Null argument at index " + i);
            }

            Py_INCREF(arg);
            PyTuple_SetItem(tuple, i, arg); // steals that reference
        }

        return tuple;
    }

    public static PyObject pyLong(long val) {
        return PyLong_FromLong(val);
    }

    public static PyObject pyDouble(double val) {
        return PyFloat_FromDouble(val);
    }

    public static PyObject pyBool(boolean val) {
        return PyBool_FromLong(val ? 1 : 0);
    }

    public static PyObject pyStr(String obj) {
        return PyUnicode_FromString(obj);
    }

    /**
     * Equivalent to Python:
     * <code>b = bytes("abc", "utf-8")</code>
     */
    public static PyObject pyBytesStr(String value) {
        if (value == null) {
            return null;
        }

        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);

        try (var ptr = new BytePointer(bytes)) {
            var pyBytes = PyBytes_FromStringAndSize(ptr, bytes.length);
            checkError();
            return pyBytes;   // new reference
        }
    }

    /**
     * Equivalent to Python:
     * <code>b = bytearray("abc", "utf-8")</code>
     */
    public static PyObject pyByteArrayStr(String value) {
        if (value == null) {
            return null;
        }

        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);

        try (var ptr = new BytePointer(bytes)) {
            var pyByteArray = PyByteArray_FromStringAndSize(ptr, bytes.length);
            checkError();
            return pyByteArray;   // new reference
        }
    }

    public static void incRef(PyObject obj) {
        Py_INCREF(obj);
    }

    public static void decRef(PyObject obj) {
        Py_DECREF(obj);
    }

    public static void refDecSafe(PyObject obj) {
        if (isPyNull(obj)) {
            throw new IllegalStateException("PyObject is null!");
        }

        long refCount = Py_REFCNT(obj);
        if (refCount <= 0) {
            throw new IllegalStateException("ActionResult already closed or invalid! RefCount: " + refCount);
        }

        Py_DECREF(obj);
    }

    public static long refCount(PyObject obj) {
        return Py_REFCNT(obj);
    }

    static void checkError() {
        try (var err = PyErr_Occurred()) {
            if (err != null) {
                PyErr_Print();
                throw new RuntimeException("Python error occurred");
            }
        }
    }

    static void printDict(PyObject dict, String msg) {
        log.info(msg);
        try (var items = PyDict_Items(dict)) {
            long size = PyList_Size(items);

            for (int i = 0; i < size; i++) {
                try (var item = PyList_GetItem(items, i);
                     var key = PyTuple_GetItem(item, 0);
                     var value = PyTuple_GetItem(item, 1)) {

                    String keyStr = toStr(key);
                    String valueStr = toStr(value);
                    IO.println("  " + keyStr + " = " + valueStr);
                }
            }
        }
    }
}
