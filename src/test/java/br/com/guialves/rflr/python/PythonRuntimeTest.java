package br.com.guialves.rflr.python;

import org.junit.jupiter.api.*;

import static br.com.guialves.rflr.python.PythonRuntime.*;
import static org.junit.jupiter.api.Assertions.*;

class PythonRuntimeTest {

    @BeforeAll
    static void setUp() {
        PythonRuntime.initPython();
        assertDoesNotThrow(() -> insideGil(() -> exec("x = 1")));
    }

    @Test
    void testExec() {
        assertDoesNotThrow(() -> exec("test_var = 42"));
        try (var result = eval("test_var")) {
            assertEquals(42L, toLong(result));
        }
    }

    @Test
    void testExecWithMultipleLines() {
        assertDoesNotThrow(() -> exec("""
        a = 10
        b = 20
        c = a + b
        """));

        try (var result = eval("c")) {
            assertEquals(30L, toLong(result));
        }
    }

    @Test
    void testExecWithError() {
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            exec("invalid syntax here %%%");
        });

        assertTrue(exception.getMessage().contains("Python error occurred"));
    }

    @Test
    void testExecIsolated() {
        assertDoesNotThrow(() -> execIsolated("isolated_var = 999"));
        assertThrows(RuntimeException.class, () -> eval("isolated_var").close());
    }

    @Test
    void testExecIsolatedDoesNotPolluteGlobals() {
        exec("global_var = 'global'");
        execIsolated("global_var = 'isolated'");
        try (var result = eval("global_var")) {
            assertEquals("global", str(result));
        }
    }

    @Test
    void testEval() {
        exec("x = 5");

        try (var result = eval("x * 2")) {
            assertEquals(10L, toLong(result));
        }
    }

    @Test
    void testEvalWithExpression() {
        try (var result = eval("2 + 2")) {
            assertEquals(4L, toLong(result));
        }
    }

    @Test
    void testEvalWithError() {
        assertThrows(RuntimeException.class, () -> {
            try (var _ = eval("undefined_variable")) {
                // Must not reach here
            }
        });
    }

    @Test
    void testPyIntArrayToJavaTuple() {
        exec("arr = (1, 2, 3, 4, 5)");

        try (var arr = eval("arr")) {
            int[] result = pyIntArrayToJava(arr);

            assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
        }
    }

    @Test
    void testPyIntArrayToJavaList() {
        exec("arr = [1, 2, 3, 4, 5]");

        try (var arr = eval("arr")) {
            int[] result = pyIntArrayToJava(arr);

            assertArrayEquals(new int[]{1, 2, 3, 4, 5}, result);
        }
    }

    @Test
    void testPyDictToJava1() {
        exec("d = {'a': 1, 'b': 2, 'c': 3}");

        try (var dict = eval("d")) {
            var result = pyDictToJava(dict);

            assertEquals(3, result.size());
            assertEquals(1L, result.get("a"));
            assertEquals(2L, result.get("b"));
            assertEquals(3L, result.get("c"));
        }
    }

    @Test
    void testPyDictToJava2() {
        exec("d = {'a': 'A', 'b': 'B', 'c': 'C'}");

        try (var dict = eval("d")) {
            var result = pyDictToJava(dict);

            assertEquals(3, result.size());
            assertEquals("A", result.get("a"));
            assertEquals("B", result.get("b"));
            assertEquals("C", result.get("c"));
        }
    }

    @Test
    void testPyDictToJava3() {
        exec("d = {1: \"A\", 2: \"B\", 3: \"C\"}");

        try (var dict = eval("d")) {
            var result = pyDictToJava(dict);

            assertEquals(3, result.size());
            assertEquals("A", result.get(1L));
            assertEquals("B", result.get(2L));
            assertEquals("C", result.get(3L));
        }
    }

    @Test
    void testStr() {
        exec("s = 'Hello, Python!'");

        try (var s = eval("s")) {
            assertEquals("Hello, Python!", str(s));
        }
    }

    @Test
    void testStrWithNone() {
        try (var none = eval("None")) {
            assertNull(str(none));
        }
    }

    @Test
    void testStrWithNull() {
        assertNull(str(null));
    }

    @Test
    void testAttr() {
        exec("""
        class MyClass:
            def __init__(self):
                self.value = 42
        pyObj = MyClass()
        """);

        try (var obj = eval("pyObj");
             var value = attr(obj, "value")) {
            assertEquals(42L, toLong(value));
        }
    }

    @Test
    void testAttrStr() {
        exec("""
        class Person:
            def __init__(self):
                self.name = "Alice"
        person = Person()
        """);

        try (var person = eval("person")) {
            assertEquals("Alice", attrStr(person, "name"));
        }
    }

    @Test
    void testToLong() {
        try (var num = eval("42")) {
            assertEquals(42L, toLong(num));
        }
    }

    @Test
    void testToDouble() {
        try (var num = eval("3.14")) {
            assertEquals(3.14, toDouble(num), 0.001);
        }
    }

    @Test
    void testToBool() {
        try (var trueVal = eval("True");
             var falseVal = eval("False")) {
            assertTrue(toBool(trueVal));
            assertFalse(toBool(falseVal));
        }
    }

    @Test
    void testCallMethod() {
        exec("""
        class Calculator:
            def add(self, a, b):
                return a + b
        calc = Calculator()
        """);

        try (var calc = eval("calc");
             var a = pyLong(10);
             var b = pyLong(20);
             var result = callMethod(calc, "add", a, b)) {
            assertEquals(30L, toLong(result));
        }
    }

    @Test
    void testCallMethodNotFound() {
        exec("""
        class MyClass:
            pass
        pyObj = MyClass()
        """);

        try (var obj = eval("pyObj")) {
            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> callMethod(obj, "non_existent_method")
            );

            assertTrue(exception.getMessage().contains("Method not found"));
        }
    }

    @Test
    void testCallFunction() {
        exec("""
        def multiply(a, b):
            return a * b
        """);

        try (var fn = eval("multiply");
             var a = pyLong(5);
             var b = pyLong(6);
             var result = callFunction(fn, a, b)) {
            assertEquals(30L, toLong(result));
        }
    }

    @Test
    void testPyLong() {
        try (var num = pyLong(12345)) {
            assertEquals(12345L, toLong(num));
        }
    }

    @Test
    void testPyDouble() {
        try (var num = pyDouble(2.71828)) {
            assertEquals(2.71828, toDouble(num), 0.00001);
        }
    }

    @Test
    void testPyBool() {
        try (var trueVal = pyBool(true);
             var falseVal = pyBool(false)) {
            assertTrue(toBool(trueVal));
            assertFalse(toBool(falseVal));
        }
    }

    @Test
    void testPyStr() {
        try (var s = pyStr("Test String")) {
            assertEquals("Test String", str(s));
        }
    }

    @Test
    void testGetItem() {
        exec("t = (10, 20, 30)");

        try (var tuple = eval("t");
             var item0 = getItem(tuple, 0);
             var item1 = getItem(tuple, 1);
             var item2 = getItem(tuple, 2)) {
            assertEquals(10L, toLong(item0));
            assertEquals(20L, toLong(item1));
            assertEquals(30L, toLong(item2));
        }
    }

    @Test
    void testRefCountManagement() {
        exec("x = [1, 2, 3]");

        try (var list = eval("x")) {
            refInc(list);
            refDec(list);
            assertDoesNotThrow(() -> str(list));
        }
    }

    @Test
    void testMultipleExecCalls() {
        for (int i = 0; i < 10; i++) {
            exec("iteration_" + i + " = " + i);
        }

        try (var result = eval("iteration_5")) {
            assertEquals(5L, toLong(result));
        }
    }

    @Test
    void testComplexDataStructure() {
        exec("""
        data = {
            'numbers': [1, 2, 3],
            'nested': {
                'value': 42
            },
            'tuple': (4, 5, 6)
        }
        """);

        try (var data = eval("data")) {
            assertNotNull(data);

            try (var numbers = eval("data['numbers']")) {
                assertNotNull(numbers);
            }
        }
    }

    @Test
    void testImportModule() {
        assertDoesNotThrow(() -> exec("import math"));

        try (var pi = eval("math.pi")) {
            assertEquals(Math.PI, toDouble(pi), 0.00001);
        }
    }

    @Test
    void testErrorRecovery() {
        assertThrows(RuntimeException.class, () -> exec("1 / 0"));

        assertDoesNotThrow(() -> exec("recovery_test = 'ok'"));

        try (var result = eval("recovery_test")) {
            assertEquals("ok", str(result));
        }
    }
}