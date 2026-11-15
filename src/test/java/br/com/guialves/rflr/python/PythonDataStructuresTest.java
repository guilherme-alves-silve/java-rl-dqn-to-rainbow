package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static br.com.guialves.rflr.python.PythonRuntime.pyLong;
import static br.com.guialves.rflr.python.PythonRuntime.pyStr;
import static br.com.guialves.rflr.python.PythonRuntime.toStr;
import static org.bytedeco.cpython.global.python.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Reference count contract of the setters used by {@link PythonDataStructures}.
 *
 * <p>Measured against CPython 3.12 (the version embedded in this project):</p>
 * <table>
 *   <caption>Who consumes the reference</caption>
 *   <tr><th>API</th><th>consumes the reference?</th></tr>
 *   <tr><td>{@code PyDict_SetItem} / {@code PyDict_SetItemString}</td>
 *       <td>no: the container increfs and the caller releases its own</td></tr>
 *   <tr><td>{@code PyDict_SetDefault}</td><td>no</td></tr>
 *   <tr><td>{@code PyList_Append} / {@code PySet_Add}</td><td>no</td></tr>
 *   <tr><td>{@code PyObject_SetItem} / {@code PyObject_SetAttr}</td><td>no</td></tr>
 *   <tr><td>{@code PyList_SetItem} / {@code PyTuple_SetItem}</td>
 *       <td>yes, and it consumes the reference even when it fails</td></tr>
 * </table>
 */
@DisplayName("PythonDataStructures: reference count contract of the setters")
class PythonDataStructuresTest {

    @BeforeAll
    static void setUp() {
        PythonRuntime.initPython();
    }

    @Test
    @DisplayName("pyDict releases the reference of the key and of the value")
    void pyDictShouldReleaseKeyAndValue() {
        var key = pyStr("key");
        var value = pyStr("value");
        assertEquals(1, Py_REFCNT(key), "the caller owns the key");
        assertEquals(1, Py_REFCNT(value), "the caller owns the value");

        try (var _ = PythonDataStructures.pyDict(
                new Object[]{"k"},
                new Object[]{"v"},
                _ -> key,
                _ -> value)) {

            // PyDict_SetItem does not consume the references: the dictionary holds
            // one and the caller held another one. pyDict released the caller side,
            // so only the reference of the dictionary is left.
            assertEquals(1, Py_REFCNT(key), "the key is referenced twice");
            assertEquals(1, Py_REFCNT(value), "the value is referenced twice");
        }
    }

    @Test
    @DisplayName("pyDictStr releases the reference of the key and of the value")
    void pyDictStrShouldReleaseKeyAndValue() {
        // A string with more than one character: single latin-1 characters are
        // cached (and the small integers are immortal in CPython 3.12), so their
        // reference count does not follow the ownership.
        var key = pyStr("the-key");
        var value = pyStr("the-value");
        assertEquals(1, Py_REFCNT(key), "the caller owns the key");
        assertEquals(1, Py_REFCNT(value), "the caller owns the value");

        try (var _ = PythonDataStructures.pyDictStr(
                new String[]{"k"},
                new Object[]{"the-value"},
                _ -> value)) {

            assertEquals(1, Py_REFCNT(key), "the key is referenced twice");
            assertEquals(1, Py_REFCNT(value), "the value is referenced twice");
        }
    }

    @Test
    @DisplayName("pyList hands the reference over to the list")
    void pyListShouldConsumeTheItem() {
        var item = pyStr("the-item");
        assertEquals(1, Py_REFCNT(item), "the caller owns the item");

        try (var _ = PythonDataStructures.pyList(i -> item, 1.0f)) {
            // PyList_SetItem consumes the reference: the list holds it and the
            // caller must not release it, otherwise it would be a double free.
            assertEquals(1, Py_REFCNT(item), "the item is referenced twice");
        }
    }

    @Test
    @DisplayName("pyTuple hands the reference over to the tuple")
    void pyTupleShouldConsumeTheItem() {
        var item = pyStr("the-item");
        assertEquals(1, Py_REFCNT(item), "the caller owns the item");

        try (var _ = PythonDataStructures.pyTuple(
                (Function<Number, PyObject>) i -> item, 1.0f)) {
            // PyTuple_SetItem consumes the reference as well.
            assertEquals(1, Py_REFCNT(item), "the item is referenced twice");
        }
    }

    @Test
    @DisplayName("pyDict stores the entries")
    void pyDictShouldStoreTheEntries() {
        try (var dict = PythonDataStructures.pyDictStr(
                new String[]{"answer"},
                new Object[]{42L},
                v -> pyLong(42L))) {

            var stored = PyDict_GetItemString(dict, "answer");
            assertTrue(stored != null && !stored.isNull(), "the entry was not stored");
            assertEquals(42L, PythonRuntime.toLong(stored));
        }
    }

    @Test
    @DisplayName("pyList stores the items")
    void pyListShouldStoreTheItems() {
        try (var list = PythonDataStructures.pyList(1.0f, 2.0f, 3.0f)) {
            assertEquals(3, PyList_Size(list));
            assertEquals("1.0", toStr(PyList_GetItem(list, 0)));
            assertEquals("3.0", toStr(PyList_GetItem(list, 2)));
        }
    }
}
