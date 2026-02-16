package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static br.com.guialves.rflr.python.PythonDataStructures.*;
import static br.com.guialves.rflr.python.PythonTypeChecks.*;
import static org.bytedeco.cpython.global.python.PyDict_SetItem;
import static org.bytedeco.cpython.global.python.PyList_Append;
import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PythonTypeChecksTest {

    @BeforeAll
    static void setUpPython() {
        initPython();
    }

    @AfterAll
    static void tearDownPython() {
        finalizePython();
    }

    @Nested
    @DisplayName("Basic Type Tests")
    class BasicTypeTests {

        @Test
        @DisplayName("Should identify boolean types correctly")
        void testBool() {
            try (PyObject pyTrue = pyBool(true);
                 PyObject pyFalse = pyBool(false);
                 PyObject pyInt = pyLong(1)) {

                assertTrue(isBool(pyTrue));
                assertTrue(isBool(pyFalse));
                assertFalse(isBool(pyInt));
                assertFalse(isBool(null));
            }
        }

        @Test
        @DisplayName("Should identify long/int types correctly")
        void testLong() {
            try (PyObject pyInt = pyLong(42);
                 PyObject pyLong = pyLong(9999999999L);
                 PyObject pyDouble = pyDouble(3.14);
                 PyObject pyBool = pyBool(true)) {

                assertTrue(isLong(pyInt));
                assertTrue(isLong(pyLong));
                assertFalse(isLong(pyDouble));
                assertTrue(isLong(pyBool)); // Booleans are subclass of int in Python
                assertFalse(isLong(null));
            }
        }

        @Test
        @DisplayName("Should identify float types correctly")
        void testFloat() {
            try (PyObject pyDouble = pyDouble(3.14159);
                 PyObject pyInt = pyLong(42);
                 PyObject pyString = pyStr("3.14")) {

                assertTrue(isDouble(pyDouble));
                assertFalse(isDouble(pyInt));
                assertFalse(isDouble(pyString));
                assertFalse(isDouble(null));
            }
        }

        @Test
        @DisplayName("Should identify complex types correctly")
        void testComplex() {
            // Create a complex number using Python execution
            exec("c = 3 + 4j");
            try (PyObject pyComplex = eval("c")) {
                try (PyObject pyDouble = pyDouble(3.14);
                     PyObject pyInt = pyLong(42)) {

                    assertTrue(isComplex(pyComplex));
                    assertFalse(isComplex(pyDouble));
                    assertFalse(isComplex(pyInt));
                    assertFalse(isComplex(null));
                }
            } finally {
                exec("del c");
            }
        }

        @Test
        @DisplayName("Should identify string types correctly")
        void testString() {
            try (PyObject pyString = pyStr("hello");
                 PyObject pyBytes = pyBytesStr("hello");
                 PyObject pyInt = pyLong(42)) {

                assertTrue(isString(pyString));
                assertFalse(isString(pyBytes));
                assertFalse(isString(pyInt));
                assertFalse(isString(null));
            }
        }

        @Test
        @DisplayName("Should identify bytes types correctly")
        void testBytes() {
            try (PyObject pyBytes = pyBytesStr("hello");
                 PyObject pyString = pyStr("hello");
                 PyObject pyByteArray = pyByteArrayStr("hello")) {

                assertTrue(isBytes(pyBytes));
                assertFalse(isBytes(pyString));
                assertFalse(isBytes(pyByteArray)); // bytearray is different from bytes
                assertFalse(isBytes(null));
            }
        }

        @Test
        @DisplayName("Should identify bytearray types correctly")
        void testByteArray() {
            try (PyObject pyByteArray = pyByteArrayStr("hello");
                 PyObject pyBytes = pyBytesStr("hello");
                 PyObject pyString = pyStr("hello")) {

                assertTrue(isByteArray(pyByteArray));
                assertFalse(isByteArray(pyBytes));
                assertFalse(isByteArray(pyString));
                assertFalse(isByteArray(null));
            }
        }
    }

    @Nested
    @DisplayName("Container Type Tests")
    class ContainerTypeTests {

        @Test
        @DisplayName("Should identify list types correctly")
        void testList() {
            try (PyObject pyList = pyList(1);
                 PyObject pyTuple = pyTuple();
                 PyObject pyDict = pyDict()) {

                // Add an element to list
                try (PyObject pyInt = pyLong(1)) {
                    incRef(pyInt);
                    PyList_Append(pyList, pyInt);
                }

                assertTrue(isList(pyList));
                assertFalse(isList(pyTuple));
                assertFalse(isList(pyDict));
                assertFalse(isList(null));
            }
        }

        @Test
        @DisplayName("Should identify tuple types correctly")
        void testTuple() {
            try (PyObject pyTuple = pyTuple();
                 PyObject pyList = pyList(1);
                 PyObject pyDict = pyDict()) {

                assertTrue(isTuple(pyTuple));
                assertFalse(isTuple(pyList));
                assertFalse(isTuple(pyDict));
                assertFalse(isTuple(null));
            }
        }

        @Test
        @DisplayName("Should identify dict types correctly")
        void testDict() {
            try (PyObject pyDict = pyDict();
                 PyObject pyList = pyList(1);
                 PyObject pySet = pySet()) {

                // Add a key-value pair
                try (PyObject pyKey = pyStr("key");
                     PyObject pyValue = pyLong(42)) {
                    incRef(pyKey);
                    incRef(pyValue);
                    PyDict_SetItem(pyDict, pyKey, pyValue);
                }

                assertTrue(isDict(pyDict));
                assertFalse(isDict(pyList));
                assertFalse(isDict(pySet));
                assertFalse(isDict(null));
            }
        }

        @Test
        @DisplayName("Should identify set types correctly")
        void testSet() {
            exec("s = {1, 2, 3}");
            try (PyObject pySet = eval("s")) {
                try (PyObject pyList = pyList(1);
                     PyObject pyFrozenSet = eval("frozenset([1,2,3])")) {

                    assertTrue(isSet(pySet));
                    assertFalse(isSet(pyList));
                    assertFalse(isSet(pyFrozenSet)); // frozenset is different from set
                    assertFalse(isSet(null));
                }
            } finally {
                exec("del s");
            }
        }

        @Test
        @DisplayName("Should identify frozenset types correctly")
        void testFrozenSet() {
            exec("fs = frozenset([1, 2, 3])");
            try (PyObject pyFrozenSet = eval("fs")) {
                try (PyObject pySet = eval("{1, 2, 3}");
                     PyObject pyList = pyList(1)) {

                    assertTrue(isFrozenSet(pyFrozenSet));
                    assertFalse(isFrozenSet(pySet));
                    assertFalse(isFrozenSet(pyList));
                    assertFalse(isFrozenSet(null));
                }
            } finally {
                exec("del fs");
            }
        }
    }

    @Nested
    @DisplayName("Iterator Type Tests")
    class IteratorTypeTests {

        @Test
        @DisplayName("Should identify list iterator types correctly")
        void testListIterator() {
            exec("my_list = [1, 2, 3]");
            exec("iter_list = iter(my_list)");

            try (PyObject pyListIterator = eval("iter_list")) {
                try (PyObject pyList = eval("my_list");
                     PyObject pyReverseIterator = eval("reversed(my_list)")) {

                    assertTrue(isListIterator(pyListIterator));
                    assertFalse(isListIterator(pyList));
                    assertFalse(isListIterator(pyReverseIterator));
                    assertFalse(isListIterator(null));
                }
            } finally {
                exec("del my_list, iter_list");
            }
        }

        @Test
        @DisplayName("Should identify reverse list iterator types correctly")
        void testReverseListIterator() {
            exec("my_list = [1, 2, 3]");
            exec("rev_iter = reversed(my_list)");

            try (PyObject pyReverseIterator = eval("rev_iter")) {
                try (PyObject pyList = eval("my_list");
                     PyObject pyListIterator = eval("iter(my_list)")) {

                    assertTrue(isReverseListIterator(pyReverseIterator));
                    assertFalse(isReverseListIterator(pyList));
                    assertFalse(isReverseListIterator(pyListIterator));
                    assertFalse(isReverseListIterator(null));
                }
            } finally {
                exec("del my_list, rev_iter");
            }
        }
    }

    @Nested
    @DisplayName("Function and Method Type Tests")
    class FunctionMethodTests {

        @Test
        @DisplayName("Should identify function types correctly")
        void testFunction() {
            exec("def test_func(): return 42");

            try (PyObject pyFunction = eval("test_func")) {
                try (PyObject pyInt = pyLong(42);
                     PyObject pyMethod = eval("dict.get")) {

                    assertTrue(isFunction(pyFunction));
                    assertFalse(isFunction(pyInt));
                    assertFalse(isFunction(pyMethod));
                    assertFalse(isFunction(null));
                }
            } finally {
                exec("del test_func");
            }
        }
    }

    @Nested
    @DisplayName("Module and Type Tests")
    class ModuleTypeTests {

        @Test
        @DisplayName("Should identify module types correctly")
        void testModule() {
            exec("import sys");

            try (PyObject pyModule = eval("sys")) {
                try (PyObject pyInt = pyLong(42);
                     PyObject pyList = pyList(1)) {

                    assertTrue(isModule(pyModule));
                    assertFalse(isModule(pyInt));
                    assertFalse(isModule(pyList));
                    assertFalse(isModule(null));
                }
            } finally {
                exec("del sys");
            }
        }

        @Test
        @DisplayName("Should identify type types correctly")
        void testType() {
            exec("class MyClass: pass");

            try (PyObject pyClass = eval("MyClass");
                 PyObject pyInstance = eval("MyClass()");
                 PyObject pyInt = pyLong(42)) {

                assertTrue(isType(pyClass));
                assertFalse(isType(pyInstance));
                assertFalse(isType(pyInt));
                assertFalse(isType(null));
            } finally {
                exec("del MyClass");
            }
        }
    }

    @Nested
    @DisplayName("Special Type Tests")
    class SpecialTypeTests {

        @Test
        @DisplayName("Should identify property types correctly")
        void testProperty() {
            exec("""
                class MyClass:
                    @property
                    def my_prop(self):
                        return 42
                """);

            try (PyObject pyProperty = eval("MyClass.my_prop")) {
                try (PyObject pyInt = pyLong(42);
                     PyObject pyMethod = eval("MyClass.__init__")) {

                    assertTrue(isProperty(pyProperty));
                    assertFalse(isProperty(pyInt));
                    assertFalse(isProperty(pyMethod));
                    assertFalse(isProperty(null));
                }
            } finally {
                exec("del MyClass");
            }
        }

        @Test
        @DisplayName("Should identify slice types correctly")
        void testSlice() {
            exec("s = slice(5)");

            try (PyObject pySlice = eval("s")) {
                try (PyObject pyInt = pyLong(42);
                     PyObject pyList = pyList(1)) {

                    assertTrue(isSlice(pySlice));
                    assertFalse(isSlice(pyInt));
                    assertFalse(isSlice(pyList));
                    assertFalse(isSlice(null));
                }
            } finally {
                exec("del s");
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCaseTests {

        @Test
        @DisplayName("Should handle null input gracefully")
        void testNullInput() {
            assertFalse(isBool(null));
            assertFalse(isLong(null));
            assertFalse(isDouble(null));
            assertFalse(isString(null));
            assertFalse(isList(null));
            assertFalse(isDict(null));
            assertFalse(isFunction(null));
            assertFalse(isModule(null));
        }

        @ParameterizedTest
        @MethodSource("provideTypesForCrossChecking")
        @DisplayName("Should correctly distinguish between different types")
        void testTypeDistinctness(PyObject testObj, String trueType, String[] falseTypes) {
            switch (trueType) {
                case "list" -> {
                    assertTrue(isList(testObj));
                    assertFalse(isTuple(testObj));
                    assertFalse(isDict(testObj));
                }
                case "tuple" -> {
                    assertTrue(isTuple(testObj));
                    assertFalse(isList(testObj));
                    assertFalse(isDict(testObj));
                }
                case "dict" -> {
                    assertTrue(isDict(testObj));
                    assertFalse(isList(testObj));
                    assertFalse(isTuple(testObj));
                }
            }
        }

        static Stream<Arguments> provideTypesForCrossChecking() {
            return Stream.of(
                    Arguments.of(createPyList(), "list", new String[]{"tuple", "dict"}),
                    Arguments.of(createPyTuple(), "tuple", new String[]{"list", "dict"}),
                    Arguments.of(createPyDict(), "dict", new String[]{"list", "tuple"})
            );
        }

        private static PyObject createPyList() {
            return pyList(1);
        }

        private static PyObject createPyTuple() {
            return pyTuple();
        }

        private static PyObject createPyDict() {
            PyObject dict = pyDict();
            try (PyObject key = pyStr("key");
                 PyObject value = pyLong(42)) {
                incRef(key);
                incRef(value);
                PyDict_SetItem(dict, key, value);
            }
            return dict;
        }
    }

    @Nested
    @DisplayName("Reference Counting Tests")
    class ReferenceCountingTests {

        @Test
        @DisplayName("Should not affect reference counts")
        void testNoRefCountChange() {
            try (var obj = pyLong(42)) {
                long initialRefCount = refCount(obj);

                for (int i = 0; i < 10; i++) {
                    isLong(obj);
                    isDouble(obj);
                    isBool(obj);
                }

                assertEquals(initialRefCount, refCount(obj),
                        "Reference count should not change when checking type");
            }
        }

        @Test
        @DisplayName("Should work with objects that have multiple references")
        void testWithMultipleReferences() {
            var obj = pyLong(42);
            incRef(obj);

            try {
                long refCount = refCount(obj);
                assertTrue(isLong(obj));
                assertEquals(refCount, refCount(obj),
                        "Reference count should remain stable");
            } finally {
                decRef(obj);
                decRef(obj);
            }
        }
    }
}
