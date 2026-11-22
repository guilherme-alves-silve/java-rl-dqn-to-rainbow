package br.com.guialves.rflr.python;

import lombok.Cleanup;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.javacpp.SizeTPointer;

import java.lang.reflect.Array;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonTypeChecks.*;
import static org.bytedeco.cpython.global.python.*;

public final class PythonDataStructures {

    private PythonDataStructures() {
        throw new IllegalArgumentException("No PythonDataStructures!");
    }

    public static PyObject getItem(PyObject obj, int pos) {
        if (!isSequence(obj)) {
            throw new IllegalArgumentException("Expected PyTuple");
        }
        var item = getSequenceItem(obj, pos);
        if (item == null) checkError();
        return PyRef.of(item); // new reference
    }

    public static boolean getItemBool(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toBool(item);
    }

    public static long getItemLong(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toLong(item);
    }

    public static double getItemDouble(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toDouble(item);
    }

    public static <K, V> Map<K, V> getItemMap(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toMap(item);
    }

    @SuppressWarnings("unchecked")
    public static <K, V> Map<K, V> toMap(PyObject obj) {

        if (obj == null || Py_IsNone(obj) >= 1) {
            return null;
        }

        if (!isDict(obj)) throw new IllegalArgumentException("Object is not a Python dict");

        var result = new LinkedHashMap<K, V>();

        try (var pos = new SizeTPointer(1);
             var keyRef = new PyObject(null);
             var valueRef = new PyObject(null)) {

            pos.put(0);

            while (PyDict_Next(obj, pos, keyRef, valueRef) != 0) {

                var javaKey = primitiveOrObjFromPy(keyRef);
                var javaValue = primitiveOrObjFromPy(valueRef);

                result.put((K) javaKey, (V) javaValue);
            }
        }

        return result;
    }

    public static double[] getItemDoubleArray(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toDoubleArray(item);
    }

    public static boolean[] getItemBoolArray(PyObject obj, int pos) {
        @Cleanup var item = getItem(obj, pos);
        return toBoolArray(item);
    }

    public static double[] toDoubleArray(PyObject obj) {
        if (isPyNull(obj)) return null;
        validateSequence(obj);
        int size = (int) getSequenceSize(obj);
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            var item = getSequenceItem(obj, i);
            try {
                if (!isDouble(item) && !isNumpyArray(item)) {
                    throw new IllegalArgumentException("element is not double");
                }
                result[i] = toDoubleChecked(item);
            } finally {
                Py_DECREF(item);
            }
        }

        return result;
    }

    public static int[] toIntArray(PyObject obj) {
        if (isPyNull(obj)) return null;
        validateSequence(obj);
        int size = (int) getSequenceSize(obj);
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            var item = getSequenceItem(obj, i);
            try {
                if (!isLong(item) && !isNumpyArray(item)) {
                    throw new IllegalArgumentException("element is not int");
                }
                result[i] = (int) toLongChecked(item);
            } finally {
                Py_DECREF(item);
            }
        }

        return result;
    }

    public static long[] toLongArray(PyObject obj) {
        if (isPyNull(obj)) return null;
        validateSequence(obj);
        int size = (int) getSequenceSize(obj);
        long[] result = new long[size];
        for (int i = 0; i < size; ++i) {
            var item = getSequenceItem(obj, i);
            try {
                result[i] = toLongChecked(item);
            } finally {
                Py_DECREF(item);
            }
        }
        return result;
    }

    /**
     * Converts a sequence of booleans into a Java {@code boolean[]}.
     * <p>
     * {@code PyObject_IsTrue()} is used instead of the string representation because it
     * works for the Python {@code bool} and for the NumPy {@code numpy.bool} scalar
     * alike, and it never leaves a pending exception behind.
     * </p>
     *
     * @param obj the sequence of booleans
     * @return the values as a Java boolean array
     */
    public static boolean[] toBoolArray(PyObject obj) {
        validateSequence(obj);
        int size = (int) getSequenceSize(obj);

        boolean[] result = new boolean[(int) size];
        for (int i = 0; i < size; ++i) {
            var item = getSequenceItem(obj, i);
            try {
                result[i] = toBool(item);
            } finally {
                Py_DECREF(item);
            }
        }
        return result;
    }

    public static <T> T[] toArray(PyObject obj, Class<T> componentType, Function<PyObject, T> mapper) {
        if (isPyNull(obj)) {
            return null;
        }

        validateSequence(obj);

        int size = (int) getSequenceSize(obj);

        @SuppressWarnings("unchecked")
        T[] result = (T[]) Array.newInstance(componentType, size);

        for (int i = 0; i < size; ++i) {
            var item = getSequenceItem(obj, i); // borrowed reference
            result[i] = mapper.apply(item);
        }

        return result;
    }

    private static void validateSequence(PyObject obj) {
        if (!isSequence(obj)) throw new IllegalArgumentException("Expected PySequence");
    }

    public static PyObject pyList(Function<Number, PyObject> mapper, Number... values) {
        if (values == null || mapper == null)
            throw new IllegalArgumentException("Mapper and values cannot be null");

        PyObject pyList = PyList_New(values.length);

        for (int i = 0; i < values.length; ++i) {
            PyObject item = mapper.apply(values[i]);
            if (item == null) throw new IllegalStateException("Mapper returned null");

            if (PyList_SetItem(pyList, i, item) != 0) {
                // PyList_SetItem consumes the reference, even when it fails, so the
                // item must never be released by the caller.
                checkError();
            }
        }
        return PyRef.of(pyList);
    }

    public static PyObject pyList(double... values) {
        return buildNumericList(values.length, i -> pyDouble(values[i]));
    }

    public static PyObject pyList(float... values) {
        return buildNumericList(values.length, i -> pyDouble(values[i]));
    }

    public static PyObject pyList(long... values) {
        return buildNumericList(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pyList(int... values) {
        return buildNumericList(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pyList(boolean... values) {
        return buildNumericList(values.length, i -> pyLong(values[i] ? 1L : 0L));
    }

    private static PyObject buildNumericList(int length, Function<Integer, PyObject> supplier) {
        PyObject pyList = PyList_New(length);

        for (int i = 0; i < length; ++i) {
            PyObject item = supplier.apply(i);

            if (PyList_SetItem(pyList, i, item) != 0) {
                // PyList_SetItem consumes the reference, even when it fails, so the
                // item must never be released by the caller.
                checkError();
            }
        }
        return PyRef.of(pyList);
    }

    public static PyObject pyTuple() {
        PyObject tuple = PyTuple_New(0);
        if (tuple == null) checkError();
        return PyRef.of(tuple);
    }

    public static PyObject pyTuple(int size) {
        if (size < 0) throw new IllegalArgumentException("Negative tuple size");
        PyObject tuple = PyTuple_New(size);
        if (tuple == null) checkError();
        return PyRef.of(tuple);
    }

    public static PyObject pyTuple(Function<Number, PyObject> mapper, Number... values) {
        return buildTuple(values.length, i -> mapper.apply(values[i]));
    }

    public static PyObject pyTuple(Function<Object, PyObject> mapper, Object... values) {
        return buildTuple(values.length, i -> mapper.apply(values[i]));
    }

    public static PyObject pyTuple(int... values) {
        return buildTuple(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pyTuple(long... values) {
        return buildTuple(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pyTuple(double... values) {
        return buildTuple(values.length, i -> pyDouble(values[i]));
    }

    public static PyObject pyTuple(String... values) {
        return buildTuple(values.length, i -> pyStr(values[i]));
    }

    private static PyObject buildTuple(int length, Function<Integer, PyObject> supplier) {
        PyObject tuple = PyTuple_New(length);
        if (tuple == null) checkError();

        for (int i = 0; i < length; ++i) {
            PyObject item = supplier.apply(i);
            if (item == null) checkError();

            if (PyTuple_SetItem(tuple, i, item) != 0) {
                // PyTuple_SetItem consumes the reference, even when it fails, so the
                // item must never be released by the caller.
                checkError();
            }
        }
        return PyRef.of(tuple);
    }

    public static PyObject pySet() {
        PyObject set = PySet_New(null);
        if (set == null) checkError();
        return PyRef.of(set);
    }

    public static PyObject pySet(PyObject iterable) {
        PyObject set = PySet_New(iterable);
        if (set == null) checkError();
        return PyRef.of(set);
    }

    public static PyObject pySet(int... values) {
        return buildSet(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pySet(long... values) {
        return buildSet(values.length, i -> pyLong(values[i]));
    }

    public static PyObject pySet(double... values) {
        return buildSet(values.length, i -> pyDouble(values[i]));
    }

    public static PyObject pySet(String... values) {
        return buildSet(values.length, i -> pyStr(values[i]));
    }

    public static PyObject pySet(Function<Number, PyObject> mapper, Number... values) {
        return buildSet(values.length, i -> mapper.apply(values[i]));
    }

    public static PyObject pySet(Function<Object, PyObject> mapper, Object... values) {
        return buildSet(values.length, i -> mapper.apply(values[i]));
    }

    private static PyObject buildSet(int length, Function<Integer, PyObject> supplier) {
        var set = pySet();
        for (int i = 0; i < length; ++i) {
            try (PyObject item = supplier.apply(i)) {
                if (item == null) checkError();
                if (PySet_Add(set, item) != 0) {
                    checkError();
                }
            }
        }

        // Already owned by PyRef
        return set;
    }

    public static PyObject pyDict() {
        PyObject dict = PyDict_New();
        if (dict == null) checkError();
        return PyRef.of(dict);
    }

    public static PyObject pyDict(Object[] keys, Object[] values,
                                  Function<Object, PyObject> keyMapper,
                                  Function<Object, PyObject> valueMapper) {

        if (keys == null || values == null ||
                keyMapper == null || valueMapper == null) {
            throw new IllegalArgumentException("Arguments cannot be null");
        }

        if (keys.length != values.length) {
            throw new IllegalArgumentException("Keys and values arrays must match");
        }

        PyObject dict = PyDict_New();
        if (dict == null) checkError();

        for (int i = 0; i < keys.length; i++) {
            var key = keyMapper.apply(keys[i]);
            var value = valueMapper.apply(values[i]);

            try {
                if (key == null || value == null)
                    throw new IllegalStateException("Mapper returned null");

                // PyDict_SetItem does NOT consume the references: the dictionary
                // stores its own reference to the key and to the value, and the
                // caller keeps its own. Both are released below.
                if (PyDict_SetItem(dict, key, value) != 0) {
                    checkError();
                }
            } finally {
                decRef(key);
                decRef(value);
            }
        }
        return PyRef.of(dict);
    }

    public static PyObject pyDictStr(String[] keys, Object[] values,
                                     Function<Object, PyObject> valueMapper) {

        if (keys == null || values == null || valueMapper == null)
            throw new IllegalArgumentException("Arguments cannot be null");

        if (keys.length != values.length)
            throw new IllegalArgumentException("Keys and values arrays must match");

        PyObject dict = PyDict_New();
        if (dict == null) checkError();

        for (int i = 0; i < keys.length; ++i) {
            var key = pyStr(keys[i]);
            var value = valueMapper.apply(values[i]);

            try {
                if (key == null || value == null)
                    throw new IllegalStateException("Mapper returned null");

                // PyDict_SetItem does NOT consume the references: the dictionary
                // stores its own reference to the key and to the value, and the
                // caller keeps its own. Both are released below.
                if (PyDict_SetItem(dict, key, value) != 0) {
                    checkError();
                }
            } finally {
                decRef(key);
                decRef(value);
            }
        }
        return PyRef.of(dict);
    }

    /**
     * Converts a Python value into a Java value.
     * <p>
     * <b>Why NumPy objects need a dedicated branch:</b> {@code PySequence_Check()} is
     * {@code true} for a {@code numpy.ndarray}, so a generic sequence conversion would
     * call {@code PySequence_GetItem()}, which returns NumPy <i>scalars</i>. Converting a
     * {@code numpy.bool} with {@code PyLong_AsLong()} raises
     * {@code TypeError: 'numpy.bool' object cannot be interpreted as an integer}
     * (NumPy 2.x removed {@code __index__} from {@code numpy.bool}) and, since the error is
     * not checked, it stays <b>pending</b> in the CPython error indicator. Any later call
     * that returns a value is then converted by CPython 3.12 into
     * {@code SystemError: ... returned a result with an exception set}.
     * </p>
     *
     * @param obj the Python object
     * @return the Java representation of the object
     */
    private static Object primitiveOrObjFromPy(PyObject obj) {

        if (obj == null || Py_IsNone(obj) >= 1) {
            return null;
        }

        // NumPy objects (ndarray and NumPy scalars) must be handled before the generic
        // sequence branch: PySequence_Check() is true for ndarray and the items are
        // NumPy scalars.
        if (isNumpyArray(obj)) {
            return numpyValue(obj);
        }

        if (isBool(obj)) {
            return PyObject_IsTrue(obj) != 0;
        }

        if (isLong(obj)) {
            return toLongChecked(obj);
        }

        if (isDouble(obj)) {
            return toDoubleChecked(obj);
        }

        if (isString(obj)) {
            return toStr(obj);
        }

        if (isDict(obj)) {
            return toMap(obj);
        }

        if (isSequence(obj)) {
            return toLongArray(obj);
        }

        throw new IllegalArgumentException(
                "Unsupported Python type in dict: " + toStr(obj)
        );
    }

    /**
     * Converts a NumPy object into a Java value, according to its dtype.
     * <p>
     * A NumPy scalar ({@code numpy.bool}, {@code numpy.int64}, {@code numpy.float64}, ...)
     * has {@code ndim == 0} and is converted into a single Java value; an {@code ndarray}
     * is converted into a Java array. NumPy objects also expose
     * {@code __array_interface__}, so {@link PythonTypeChecks#isNumpyArray(PyObject)} is
     * {@code true} for both.
     * </p>
     *
     * @param obj the NumPy object
     * @return {@code Boolean}, {@code Long}, {@code Double}, {@code boolean[]},
     * {@code long[]} or {@code double[]}
     */
    private static Object numpyValue(PyObject obj) {
        var dtype = numpyDtype(obj);

        if (numpyNdim(obj) == 0) {
            // NumPy scalar.
            return switch (dtype) {
                case "bool" -> toBool(obj);
                case "int8", "int16", "int32", "int64",
                     "uint8", "uint16", "uint32", "uint64" -> toLongChecked(obj);
                case "float16", "float32", "float64" -> toDoubleChecked(obj);
                default -> throw unsupportedNumpyDtype(dtype);
            };
        }

        return switch (dtype) {
            case "bool" -> toBoolArray(obj);
            case "int8", "int16", "int32", "int64",
                 "uint8", "uint16", "uint32", "uint64" -> toLongArray(obj);
            case "float16", "float32", "float64" -> toDoubleArray(obj);
            default -> throw unsupportedNumpyDtype(dtype);
        };
    }

    /**
     * Reads the dtype name of a NumPy object (e.g. {@code bool}, {@code int64},
     * {@code float64}).
     *
     * @param obj the NumPy object
     * @return the dtype name
     */
    private static String numpyDtype(PyObject obj) {
        var dtype = attr(obj, "dtype");
        try {
            return attrStr(dtype, "name");
        } finally {
            decRef(dtype);
        }
    }

    /**
     * Reads the number of dimensions of a NumPy object: {@code 0} for a NumPy scalar and
     * {@code >= 1} for an {@code ndarray}.
     *
     * @param obj the NumPy object
     * @return the number of dimensions
     */
    private static int numpyNdim(PyObject obj) {
        var ndim = attr(obj, "ndim");
        try {
            return (int) toLongChecked(ndim);
        } finally {
            decRef(ndim);
        }
    }

    private static IllegalArgumentException unsupportedNumpyDtype(String dtype) {
        return new IllegalArgumentException("Unsupported NumPy dtype in dict: " + dtype);
    }

    /**
     * Converts a Python integer into a Java long.
     * <p>
     * {@code PyLong_AsLong()} returns {@code -1} on error <b>and leaves the exception
     * pending</b>. This helper detects the pending error and drains it through
     * {@link PythonRuntime#checkError()}, so the error indicator is never left dirty.
     * </p>
     *
     * @param obj the Python integer
     * @return the value as a Java long
     */
    private static long toLongChecked(PyObject obj) {
        var value = PyLong_AsLong(obj);
        if (value == -1 && hasPendingError()) {
            checkError();
        }
        return value;
    }

    /**
     * Converts a Python float into a Java double, draining a pending error instead of
     * leaving it in the CPython error indicator.
     *
     * @param obj the Python float
     * @return the value as a Java double
     */
    private static double toDoubleChecked(PyObject obj) {
        var value = PyFloat_AsDouble(obj);
        if (value == -1.0 && hasPendingError()) {
            checkError();
        }
        return value;
    }

    /**
     * Tells whether the CPython error indicator holds a pending exception.
     *
     * @return {@code true} when an exception is pending
     */
    private static boolean hasPendingError() {
        var err = PyErr_Occurred();
        return err != null && !err.isNull();
    }

    public static long getSequenceSize(PyObject obj) {

        if (isList(obj)) {
            return PyList_Size(obj);
        }

        if (isTuple(obj)) {
            return PyTuple_Size(obj);
        }

        return PySequence_Size(obj);
    }

    private static PyObject getSequenceItem(PyObject obj, int i) {

        PyObject item;

        if (isList(obj)) {
            item = PyList_GetItem(obj, i);  // borrowed
            Py_INCREF(item);                // promote to new reference
            return item;
        }

        if (isTuple(obj)) {
            item = PyTuple_GetItem(obj, i); // borrowed
            Py_INCREF(item);                // promote to new reference
            return item;
        }

        // already new reference
        return PySequence_GetItem(obj, i);
    }
}
