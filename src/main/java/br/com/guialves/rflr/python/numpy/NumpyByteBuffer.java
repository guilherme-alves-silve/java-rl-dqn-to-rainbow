package br.com.guialves.rflr.python.numpy;

import br.com.guialves.rflr.python.PythonDataStructures;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.Py_buffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static org.bytedeco.cpython.global.python.*;

public class NumpyByteBuffer {

    private static final ByteOrder BYTE_ORDER;

    static {
        initPython();

        var gstate = PyGILState_Ensure();
        try {
            exec("import numpy as np");
            exec("_test_arr = np.array([1], dtype=np.float32)");

            try (var testArr = eval("_test_arr")) {
                String byteOrder = attrStr(attr(testArr, "dtype"), "byteorder");
                BYTE_ORDER = switch (byteOrder) {
                    case ">" -> ByteOrder.BIG_ENDIAN;
                    case "<" -> ByteOrder.LITTLE_ENDIAN;
                    default -> ByteOrder.nativeOrder();
                };
            }

            exec("del _test_arr");
        } finally {
            PyGILState_Release(gstate);
        }
    }

    private NumpyByteBuffer() {
        throw new IllegalArgumentException("No SafeByteBuffer!");
    }

    /**
     * Creates a ByteBuffer compatible with NumPy arrays.
     * The byte order is automatically detected and cached on first use.
     *
     * @param size buffer size in bytes
     * @return ByteBuffer configured with the correct byte order for NumPy
     */
    public static ByteBuffer onHeapBufferNumpy(int size) {
        return ByteBuffer.allocate(size).order(BYTE_ORDER);
    }

    /**
     * Creates a DirectByteBuffer (outside of heap) compatible with NumPy arrays.
     * The byte order is automatically detected and cached on first use.
     *
     * @param size buffer size in bytes
     * @return ByteBuffer configured with the correct byte order for NumPy
     */
    public static ByteBuffer offHeapBufferNumpy(int size) {
        return ByteBuffer.allocateDirect(size).order(BYTE_ORDER);
    }

    /**
     * Creates a ByteBuffer compatible with a specific NumPy array.
     * Checks the byte order of the provided array.
     * NumPy byte order indicators:
     *  '>' = big-endian
     *  '<' = little-endian
     *  '=' = native
     *  '|' = not applicable (for single-byte types)
     * @param ndarray reference NumPy array
     * @param size buffer size in bytes
     * @return ByteBuffer configured with the array's byte order
     */
    public static ByteBuffer onHeapBufferNumpy(PyObject ndarray, int size) {
        String byteOrder = attrStr(attr(ndarray, "dtype"), "byteorder");
        var order = switch (byteOrder) {
            case ">" -> ByteOrder.BIG_ENDIAN;
            case "<" -> ByteOrder.LITTLE_ENDIAN;
            case "=", "|" -> ByteOrder.nativeOrder();
            default -> BYTE_ORDER;
        };

        return ByteBuffer.allocate(size).order(order);
    }

    /**
     * Copies the raw byte contents of a NumPy {@code ndarray} into a Java {@link ByteBuffer}
     * using the CPython Buffer Protocol (PEP 3118).
     *
     * <p>This method accesses the underlying contiguous memory of the given {@code ndarray}
     * via {@code PyObject_GetBuffer} with {@code PyBUF_SIMPLE}, which guarantees a
     * pointer-to-bytes view ({@code void* + length}) without creating intermediate Python
     * objects or performing implicit copies.</p>
     *
     * <p>The size returned by {@link Py_buffer#len()} represents the exact number of bytes
     * exposed by the NumPy array (equivalent to {@code ndarray.nbytes} in Python),
     * <strong>not</strong> the number of elements.</p>
     *
     * <p><strong>Requirements:</strong>
     * <ul>
     *   <li>The NumPy array must be C-contiguous.</li>
     *   <li>The destination {@link ByteBuffer} must have sufficient capacity.</li>
     * </ul>
     * If the array is not contiguous (e.g. a slice or view with strides),
     * {@code PyObject_GetBuffer} will fail.</p>
     *
     * <p>The buffer view is always released via {@code PyBuffer_Release} to avoid
     * memory leaks and reference mismanagement.</p>
     *
     * <p>References:
     * <ul>
     *   <li>CPython C API – Buffer Protocol:
     *       <a href="https://docs.python.org/3/c-api/buffer.html">...</a></li>
     *   <li>PEP 3118 – Revising the Buffer Protocol:
     *       <a href="https://peps.python.org/pep-3118/">...</a></li>
     * </ul>
     * </p>
     *
     * @param ndarray a NumPy {@code ndarray} exposing the buffer protocol
     * @param buffer  the destination {@link ByteBuffer} to receive the raw bytes
     * @throws IllegalStateException if the array does not expose a contiguous buffer
     * @throws IllegalArgumentException if the destination buffer capacity is insufficient
     */
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

    public static int[] toIntArray(PyObject obj) {

        if (!hasAttr(obj, "dtype")) {
            return PythonDataStructures.toIntArray(obj);
        }

        String dtype = attrStr(attr(obj, "dtype"), "name");

        if (!"int32".equals(dtype)) {
            throw new IllegalArgumentException("Expected numpy int32 array, got " + dtype);
        }

        int bytes = nbytes(obj);

        var buffer = onHeapBufferNumpy(obj, bytes);
        fillFromNumpy(obj, buffer);

        var intView = buffer.asIntBuffer();

        int[] result = new int[intView.remaining()];
        intView.get(result);

        return result;
    }

    public static long[] toLongArray(PyObject obj) {

        if (!hasAttr(obj, "dtype")) {
            return PythonDataStructures.toLongArray(obj);
        }

        String dtype = attrStr(attr(obj, "dtype"), "name");

        if (!"int64".equals(dtype)) {
            throw new IllegalArgumentException("Expected numpy int64 array, got " + dtype);
        }

        int bytes = nbytes(obj);

        var buffer = onHeapBufferNumpy(obj, bytes);
        fillFromNumpy(obj, buffer);

        var longView = buffer.asLongBuffer();

        long[] result = new long[longView.remaining()];
        longView.get(result);

        return result;
    }

    public static double[] toDoubleArray(PyObject obj) {

        if (!hasAttr(obj, "dtype")) {
            return PythonDataStructures.toDoubleArray(obj);
        }

        String dtype = attrStr(attr(obj, "dtype"), "name");

        if (!("float64".equals(dtype) || "float32".equals(dtype))) {
            throw new IllegalArgumentException("Expected numpy float64 or float32 array, got " + dtype);
        }

        int bytes = nbytes(obj);

        var buffer = onHeapBufferNumpy(obj, bytes);
        fillFromNumpy(obj, buffer);

        if ("float64".equals(dtype)) {
            var doubleView = buffer.asDoubleBuffer();
            double[] result = new double[doubleView.remaining()];
            doubleView.get(result);
            return result;
        }

        // maybe separate float32 and float64 if performance demands
        var floatView = buffer.asFloatBuffer();
        int size = floatView.remaining();
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = floatView.get();
        }
        return result;
    }

    private static int nbytes(PyObject ndarray) {
        try (var nb = attr(ndarray, "nbytes")) {
            return (int) PyLong_AsLong(nb);
        }
    }
}
