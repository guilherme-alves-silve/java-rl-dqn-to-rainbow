package br.com.guialves.rflr.python;

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
        if (!isTuple(obj)) {
            throw new IllegalArgumentException("Expected PyTuple");
        }
        PyObject item = PyTuple_GetItem(obj, pos);
        if (item == null) checkError();
        return item; // borrowed reference
    }

    public static boolean getItemBool(PyObject obj, int pos) {
        return toBool(getItem(obj, pos));
    }

    public static long getItemLong(PyObject obj, int pos) {
        return toLong(getItem(obj, pos));
    }

    public static double getItemDouble(PyObject obj, int pos) {
        return toDouble(getItem(obj, pos));
    }

    public static <K, V> Map<K, V> getItemMap(PyObject obj, int pos) {
        return toMap(getItem(obj, pos));
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

                var javaKey = primitiveFromPy(keyRef);
                var javaValue = primitiveFromPy(valueRef);

                result.put((K) javaKey, (V) javaValue);
            }
        }

        return result;
    }

    public static double[] toDoubleArray(PyObject obj) {
        if (isPyNull(obj)) return null;
        validateSequence(obj);
        int size = (int) getSequenceSize(obj);
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            var item = getSequenceItem(obj, i);
            try {
                if (!isDouble(item)) throw new IllegalArgumentException("Shape element is not int");
                result[i] = PyFloat_AsDouble(item);
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
                if (!isLong(item)) throw new IllegalArgumentException("Shape element is not int");
                result[i] = (int) PyLong_AsLong(item);
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
        for (int i = 0; i < size; i++) {
            var item = getSequenceItem(obj, i);
            try {
                if (!isLong(item)) throw new IllegalArgumentException("Shape element is not int");
                result[i] = PyLong_AsLong(item);
            } finally {
                Py_DECREF(item);
            }
        }
        return result;
    }

    public static boolean[] toBoolArray(PyObject obj) {
        validateSequence(obj);
        long size = PyList_Size(obj);

        boolean[] result = new boolean[(int) size];

        for (int i = 0; i < size; i++) {
            PyObject item = PyList_GetItem(obj, i);
            long value = PyLong_AsLong(item);
            result[i] = value != 0;
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

        for (int i = 0; i < size; i++) {
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

        for (int i = 0; i < values.length; i++) {
            PyObject item = mapper.apply(values[i]);
            if (item == null) throw new IllegalStateException("Mapper returned null");

            if (PyList_SetItem(pyList, i, item) != 0) {
                item.close();
            }
        }
        return pyList;
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

        for (int i = 0; i < length; i++) {
            PyObject item = supplier.apply(i);

            if (PyList_SetItem(pyList, i, item) != 0) {
                item.close();
            }
        }
        return pyList;
    }

    public static PyObject pyTuple() {
        PyObject tuple = PyTuple_New(0);
        if (tuple == null) checkError();
        return tuple;
    }

    public static PyObject pyTuple(int size) {
        if (size < 0) throw new IllegalArgumentException("Negative tuple size");
        PyObject tuple = PyTuple_New(size);
        if (tuple == null) checkError();
        return tuple;
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

        for (int i = 0; i < length; i++) {
            PyObject item = supplier.apply(i);
            if (item == null) checkError();

            if (PyTuple_SetItem(tuple, i, item) != 0) {
                item.close();
            }
        }
        return tuple;
    }

    public static PyObject pySet() {
        PyObject set = PySet_New(null);
        if (set == null) checkError();
        return set;
    }

    public static PyObject pySet(PyObject iterable) {
        PyObject set = PySet_New(iterable);
        if (set == null) checkError();
        return set;
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
        for (int i = 0; i < length; i++) {
            try (PyObject item = supplier.apply(i)) {
                if (item == null) checkError();
                if (PySet_Add(set, item) != 0) {
                    checkError();
                }
            }
        }

        return set;
    }

    public static PyObject pyDict() {
        PyObject dict = PyDict_New();
        if (dict == null) checkError();
        return dict;
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
            try (PyObject key = keyMapper.apply(keys[i]);
                 PyObject value = valueMapper.apply(values[i])) {

                if (key == null || value == null)
                    throw new IllegalStateException("Mapper returned null");

                if (PyDict_SetItem(dict, key, value) != 0) {
                    checkError();
                }
            }
        }
        return dict;
    }

    public static PyObject pyDictStr(String[] keys, Object[] values,
                                     Function<Object, PyObject> valueMapper) {

        if (keys == null || values == null || valueMapper == null)
            throw new IllegalArgumentException("Arguments cannot be null");

        if (keys.length != values.length)
            throw new IllegalArgumentException("Keys and values arrays must match");

        PyObject dict = PyDict_New();
        if (dict == null) checkError();

        for (int i = 0; i < keys.length; i++) {
            try (PyObject key = pyStr(keys[i]);
                 PyObject value = valueMapper.apply(values[i])) {

                if (key == null || value == null)
                    throw new IllegalStateException("Mapper returned null");

                if (PyDict_SetItem(dict, key, value) != 0) {
                    checkError();
                }
            }
        }
        return dict;
    }

    private static Object primitiveFromPy(PyObject obj) {

        if (obj == null || Py_IsNone(obj) >= 1) {
            return null;
        }

        if (isBool(obj)) {
            return PyObject_IsTrue(obj) != 0;
        }

        if (isLong(obj)) {
            return PyLong_AsLong(obj);
        }

        if (isDouble(obj)) {
            return PyFloat_AsDouble(obj);
        }

        if (isString(obj)) {
            return toStr(obj);
        }

        throw new IllegalArgumentException(
                "Unsupported Python type in dict: " + toStr(obj)
        );
    }

    private static long getSequenceSize(PyObject obj) {

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
