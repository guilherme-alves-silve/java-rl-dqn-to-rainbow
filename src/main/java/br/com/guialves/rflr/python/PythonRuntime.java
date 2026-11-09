package br.com.guialves.rflr.python;

import io.vavr.CheckedFunction0;
import io.vavr.CheckedRunnable;
import lombok.Cleanup;
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

    /**
     * Evaluates a Python expression in the runtime globals.
     * <p>{@code PyRun_StringFlags} returns a <b>new</b> reference. The
     * {@code try-with-resources} block below does <b>not</b> release it (see
     * {@link #decRef(PyObject)}), so the returned object stays valid and the
     * caller owns the reference: it must be released with
     * {@link #decRef(PyObject)} or {@link #decRef(PyObject)}.</p>
     *
     * @param expression the expression to evaluate
     * @return the resulting object, as a new reference owned by the caller
     */
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
            return PyRef.of(result);
        }
    }

    /**
     * @return the number of live Python objects, as reported by
     *         {@code sys.getallocatedblocks()}
     */
    public static long allocatedPythonBlocks() {
        // eval() transfers the ownership of a new Python reference: it must be released.
        @Cleanup var value = eval("__import__('sys').getallocatedblocks()");
        return toLong(value);
    }

    public static String toType(PyObject obj) {
        var type = PyObject_Type(obj);
        try {
            return toStr(type);
        } finally {
            decRef(type);
        }
    }

    public static String toStr(PyObject obj) {
        return toStr(obj, "");
    }

    /**
     * Converts a Python object to its {@code str} representation.
     * <p><b>Reference ownership:</b> {@code PyObject_Str} and
     * {@code PyUnicode_AsUTF8String} both return a <i>new</i> reference, so they
     * are released here. {@code PyBytes_AsString} returns a <i>borrowed</i>
     * reference and must not be released.</p>
     * <p>Note that {@code try-with-resources} cannot be used to release a
     * {@link PyObject}: the JavaCPP binding declares {@code PyObject} as
     * {@code @Opaque} without a deallocator, therefore {@code close()} is a no-op
     * and never calls {@code Py_DECREF}. See {@link #decRef(PyObject)}.</p>
     */
    public static String toStr(PyObject obj, String msg) {
        if (null == obj || Py_IsNone(obj) == 1) {
            return null;
        }

        var pyStr = PyObject_Str(obj);
        try {
            if (pyStr == null) {
                PyErr_Print();
                throw new RuntimeException("PyObject_Str failed: " + msg);
            }

            var bytes = PyUnicode_AsUTF8String(pyStr);
            try {
                if (bytes == null) {
                    PyErr_Print();
                    throw new RuntimeException("PyUnicode_AsUTF8String failed: " + msg);
                }

                try (var temp = PyBytes_AsString(bytes)) {
                    return temp.getString();
                }
            } finally {
                decRef(bytes);
            }
        } finally {
            decRef(pyStr);
        }
    }

    /**
     * Reads an attribute from a Python object.
     *
     * @param obj  the object to read from
     * @param attr the attribute name
     * @return the attribute value, as a <b>new</b> Python reference owned by the
     *         caller. The caller <b>must</b> release it with {@link #decRef(PyObject)}
     *         (or {@link #decRef(PyObject)}) when done.
     */
    public static PyObject attr(PyObject obj, String attr) {
        var result = PyObject_GetAttrString(obj, attr);
        if (isPyNull(result)) {
            PyErr_Print();
            throw new IllegalArgumentException("Attribute not found: " + attr);
        }
        return result;
    }

    public static boolean hasAttr(PyObject obj, String name) {
        return PyObject_HasAttrString(obj, name) == 1;
    }

    /**
     * Reads an attribute and converts it to its {@code str} representation.
     * <p>{@code PyObject_GetAttrString} returns a <b>new</b> reference, released
     * here. See {@link #toStr(PyObject, String)} for why {@code try-with-resources}
     * cannot be used for this purpose.</p>
     */
    public static String attrStr(PyObject obj, String attr) {
        var attrObj = PyObject_GetAttrString(obj, attr);
        try {
            return toStr(attrObj);
        } finally {
            decRef(attrObj);
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
        return PyRef.of(result);
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
                return PyRef.of(result);
            } finally {
                Py_DECREF(tuple);
            }
        } finally {
            if (fn != null && !fn.isNull()) {
                Py_DECREF(fn);
            }
        }
    }

    /**
     * Calls a Python callable.
     * <p>{@code PyObject_CallObject} returns a <b>new</b> reference. The
     * {@code retainReference()} call is a no-op for {@link PyObject} (it has no
     * deallocator), so the caller owns the reference and must release it with
     * {@link #decRef(PyObject)} or {@link #decRef(PyObject)}.</p>
     *
     * @param fn   the callable
     * @param args the call arguments
     * @return the call result, as a new reference owned by the caller
     */
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
            return PyRef.of(result);
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
        if (obj != null && !obj.isNull()) {
            Py_INCREF(obj);
        }
    }

    /**
     * Releases a Python reference previously obtained from an API that returns a
     * <b>new</b> reference ({@code PyObject_GetAttrString}, {@code PyObject_CallObject},
     * {@code PyObject_Str}, {@code PyObject_Type}, {@code PyRun_StringFlags}, ...).
     *
     * <p><b>Why not {@code close()}/{@code @Cleanup}?</b> The JavaCPP binding declares
     * {@link PyObject} as {@code @Opaque} <b>without a deallocator</b>. Therefore
     * {@code Pointer.close()} (which delegates to {@code releaseReference()}) and
     * {@code retainReference()} are both no-ops for {@code PyObject}: they neither
     * call {@code Py_DECREF} nor invalidate the pointer. Every {@code try-with-resources}
     * over a {@code PyObject} silently leaks one Python reference.</p>
     *
     * <p>This method is null-safe and never touches a borrowed reference: it must only
     * be used for references whose ownership was transferred to Java.</p>
     *
     * @param obj the reference to release, may be {@code null}
     */
    public static void decRef(PyObject obj) {
        if (obj != null && !obj.isNull()) {
            Py_DECREF(obj);
        }
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
        // PyDict_Items() returns a new reference; PyList_GetItem()/PyTuple_GetItem()
        // return borrowed ones.
        var items = PyDict_Items(dict);
        try {
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
        } finally {
            decRef(items);
        }
    }
}
