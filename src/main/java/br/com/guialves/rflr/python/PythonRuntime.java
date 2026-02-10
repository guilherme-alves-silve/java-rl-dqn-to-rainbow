package br.com.guialves.rflr.python;

import lombok.SneakyThrows;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.Py_buffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;

import static org.bytedeco.cpython.global.python.*;

public final class PythonRuntime {

    public static final String JAVA_RL_SITE_PACKAGES = "JAVA_RL_SITE_PACKAGES";
    public static final String JAVA_RL_INCLUDE = "JAVA_RL_INCLUDE";
    private static boolean initialized = false;
    private static PyObject globals;

    private PythonRuntime() {
        throw new IllegalArgumentException("No PythonRuntime!");
    }

    private static File[] cachePackages() throws IOException {
        var path = org.bytedeco.cpython.presets.python.cachePackages();
        path = Arrays.copyOf(path, path.length + 2);

        var sitePackages = System.getProperty(JAVA_RL_SITE_PACKAGES,
                System.getenv(JAVA_RL_SITE_PACKAGES));
        var include = System.getProperty(JAVA_RL_INCLUDE,
                System.getenv(JAVA_RL_INCLUDE));

        if (null == sitePackages || null == include) {
            throw new IllegalStateException("It's mandatory to pass sitePackages and include");
        }

        path[path.length - 1] = new File(sitePackages);
        path[path.length - 2] = new File(include);
        return path;
    }

    @SneakyThrows
    public static synchronized void init() {
        if (initialized) return;

        System.setProperty("org.bytedeco.openblas.load", "mkl");

        initialized = Py_Initialize(cachePackages());
        globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        if (!initialized) {
            throw new IllegalArgumentException("PythonRuntime is not initialized!");
        }
    }

    public static synchronized void finalizePython() {
        if (!initialized) return;
        Py_Finalize();
        initialized = false;
        globals.close();
        globals = null;
    }

    public static void exec(String code) {
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

    public static PyObject eval(String expression) {
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

    public static void fillFromNumpy(PyObject ndarray, ByteBuffer buffer) {
        buffer.clear();
        var view = new Py_buffer();
        try {
            int rc = PyObject_GetBuffer(ndarray, view, PyBUF_SIMPLE);
            if (rc != 0) {
                throw new IllegalStateException("PyObject_GetBuffer failed (array not contiguous?)");
            }

            long size = view.len();
            if (size > buffer.capacity()) {
                throw new IllegalArgumentException(
                        "Buffer too small: capacity=" + buffer.capacity() + ", required=" + size
                );
            }
            var src = view.buf().capacity(size).asByteBuffer();
            buffer.put(src);
        } finally {
            PyBuffer_Release(view);
        }

        buffer.flip();
    }

    public static int[] pyIntArrayToJava(PyObject obj) {
        try (var pyStr = PyObject_Str(obj)) {
            var repr = PythonRuntime.str(pyStr);
            return PythonSequenceConverter.parsePythonIntArray(repr);
        }
    }

    public static Map<String, Object> pyDictToJava(PyObject obj) {
        try (var pyStr = PyObject_Str(obj)) {
            var repr = PythonRuntime.str(pyStr);
            return PythonDictConverter.parsePythonDictRepr(repr);
        }
    }


    public static String str(PyObject obj) {
        if (obj == null || Py_IsNone(obj) == 1) {
            return null;
        }

        try (var pyStr = PyObject_Str(obj)) {
            if (pyStr == null) {
                PyErr_Print();
                throw new RuntimeException("PyObject_Str failed");
            }

            try (var bytes = PyUnicode_AsUTF8String(pyStr)) {
                if (bytes == null) {
                    PyErr_Print();
                    throw new RuntimeException("PyUnicode_AsUTF8String failed");
                }

                try (var temp = PyBytes_AsString(bytes)) {
                    return temp.getString();
                }
            }
        }
    }

    public static PyObject attr(PyObject obj, String attr) {
        return PyObject_GetAttrString(obj, attr);
    }

    public static String attrStr(PyObject obj, String attr) {
        try (var attrObj = PyObject_GetAttrString(obj, attr)) {
            return str(attrObj);
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

    public static PyObject callObj(PyObject obj, String method) {
        return PyObject_CallObject(obj, null);
    }

    public static PyObject callMethod(PyObject obj, String method) {
        return PyObject_CallMethod(obj, method, null);
    }

    public static PyObject getItem(PyObject obj, int pos) {
        return PyTuple_GetItem(obj, pos);
    }

    public static PyObject callMethod(
            PyObject obj,
            String method,
            PyObject... args
    ) {
        try (var fn = PyObject_GetAttrString(obj, method)) {
            if (fn == null) {
                throw new IllegalStateException("Method not found: " + method);
            }
            return callFunction(fn, args);
        }
    }

    public static PyObject newArgs(PyObject... args) {
        var tuple = PyTuple_New(args.length);
        for (int i = 0; i < args.length; i++) {
            PyTuple_SetItem(tuple, i, args[i]);
        }
        return tuple;
    }

    public static PyObject callFunction(
            PyObject fn,
            PyObject... args
    ) {
        try (var tuple = newArgs(args)) {
            return PyObject_CallObject(fn, tuple);
        }
    }

    public static PyObject pyLong(long v) {
        return PyLong_FromLong(v);
    }

    public static PyObject pyFloat(double v) {
        return PyFloat_FromDouble(v);
    }

    public static PyObject pyBool(boolean v) {
        return PyBool_FromLong(v ? 1 : 0);
    }

    public static PyObject pyStr(String s) {
        return PyUnicode_FromString(s);
    }

    private static void checkError() {
        try (var err = PyErr_Occurred()) {
            if (err != null) {
                PyErr_Print();
                throw new RuntimeException("Python error occurred");
            }
        }
    }
}
