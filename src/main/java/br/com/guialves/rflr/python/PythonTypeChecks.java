package br.com.guialves.rflr.python;

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
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyBool_Type()) != 0;
    }

    public static boolean isLong(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyLong_Type()) != 0;
    }

    public static boolean isDouble(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyFloat_Type()) != 0;
    }

    public static boolean isComplex(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyComplex_Type()) != 0;
    }

    public static boolean isString(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyUnicode_Type()) != 0;
    }

    public static boolean isBytes(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyBytes_Type()) != 0;
    }

    public static boolean isByteArray(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyByteArray_Type()) != 0;
    }

    public static boolean isList(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyList_Type()) != 0;
    }

    public static boolean isSequence(PyObject obj) {
        return !isPyNull(obj) && PySequence_Check(obj) != 0;
    }

    public static boolean isTuple(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyTuple_Type()) != 0;
    }

    public static boolean isDict(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyDict_Type()) != 0;
    }

    public static boolean isSet(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PySet_Type()) != 0;
    }

    public static boolean isFrozenSet(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyFrozenSet_Type()) != 0;
    }

    public static boolean isListIterator(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyListIter_Type()) != 0;
    }

    public static boolean isReverseListIterator(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyListRevIter_Type()) != 0;
    }

    public static boolean isFunction(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyFunction_Type()) != 0;
    }

    public static boolean isMethod(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyMethod_Type()) != 0;
    }

    public static boolean isModule(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyModule_Type()) != 0;
    }

    public static boolean isType(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyType_Type()) != 0;
    }

    public static boolean isProperty(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PyProperty_Type()) != 0;
    }

    public static boolean isSlice(PyObject obj) {
        return !isPyNull(obj) && PyObject_TypeCheck(obj, PySlice_Type()) != 0;
    }

    public static boolean isNumpyArray(PyObject obj) {
        return obj != null &&
                PyObject_HasAttrString(obj, "__array_interface__") != 0;
    }
}
