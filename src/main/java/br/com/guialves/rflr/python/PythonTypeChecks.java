package br.com.guialves.rflr.python;

import lombok.Cleanup;
import org.bytedeco.cpython.PyObject;

import static org.bytedeco.cpython.global.python.*;

public final class PythonTypeChecks {

    private PythonTypeChecks() {
        throw new IllegalArgumentException("No PythonTypeChecks!");
    }

    public static boolean isPyNull(PyObject obj) {
        return obj == null || Py_IsNone(obj) != 0;
    }

    public static boolean isBool(PyObject obj) {
        @Cleanup var type = PyBool_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isLong(PyObject obj) {
        @Cleanup var type = PyLong_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isDouble(PyObject obj) {
        @Cleanup var type = PyFloat_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isComplex(PyObject obj) {
        @Cleanup var type = PyComplex_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isString(PyObject obj) {
        @Cleanup var type = PyUnicode_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isBytes(PyObject obj) {
        @Cleanup var type = PyBytes_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isByteArray(PyObject obj) {
        @Cleanup var type = PyByteArray_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isList(PyObject obj) {
        @Cleanup var type = PyList_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isSequence(PyObject obj) {
        return !isPyNull(obj) && PySequence_Check(obj) != 0;
    }

    public static boolean isTuple(PyObject obj) {
        @Cleanup var type = PyTuple_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isDict(PyObject obj) {
        @Cleanup var type = PyDict_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isSet(PyObject obj) {
        @Cleanup var type = PySet_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isFrozenSet(PyObject obj) {
        @Cleanup var type = PyFrozenSet_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isListIterator(PyObject obj) {
        @Cleanup var type = PyListIter_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isReverseListIterator(PyObject obj) {
        @Cleanup var type = PyListRevIter_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isFunction(PyObject obj) {
        @Cleanup var type = PyFunction_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isMethod(PyObject obj) {
        @Cleanup var type = PyMethod_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isModule(PyObject obj) {
        @Cleanup var type = PyModule_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isType(PyObject obj) {
        @Cleanup var type = PyType_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isProperty(PyObject obj) {
        @Cleanup var type = PyProperty_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isSlice(PyObject obj) {
        @Cleanup var type = PySlice_Type();
        return !isPyNull(obj) && PyObject_TypeCheck(obj, type) != 0;
    }

    public static boolean isNumpyArray(PyObject obj) {
        return obj != null &&
                PyObject_HasAttrString(obj, "__array_interface__") != 0;
    }
}
