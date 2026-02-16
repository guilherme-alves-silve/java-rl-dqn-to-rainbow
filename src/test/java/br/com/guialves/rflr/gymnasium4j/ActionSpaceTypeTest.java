package br.com.guialves.rflr.gymnasium4j;

import org.bytedeco.cpython.PyObject;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.*;
import static br.com.guialves.rflr.python.PythonRuntime.*;
import static org.junit.jupiter.api.Assertions.*;

class ActionSpaceTypeTest {

    @BeforeAll
    static void setUpPython() {
        initPython();
    }

    @AfterAll
    static void tearDownPython() {
        finalizePython();
    }

    @Nested
    @DisplayName("DISCRETE Action Space Tests")
    class DiscreteTests {

        @Test
        @DisplayName("Should create ActionResult from int value")
        void testDiscreteFromInt() {
            try (var action = DISCRETE.get(5)) {
                assertNotNull(action);
                assertEquals(DISCRETE, action.spaceType());
                assertFalse(action.isClosed());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should create ActionResult from long value")
        void testDiscreteFromLong() {
            try (var action = DISCRETE.get(100L)) {
                assertNotNull(action);
                assertEquals(DISCRETE, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should throw exception for unsupported double")
        void testDiscreteDoesNotSupportDouble() {
            assertThrows(UnsupportedOperationException.class, () -> DISCRETE.get(1.5));
        }

        @Test
        @DisplayName("Should throw exception for unsupported array")
        void testDiscreteDoesNotSupportArray() {
            assertThrows(UnsupportedOperationException.class, () -> DISCRETE.get(new int[]{1, 2, 3}));
        }

        @Test
        @DisplayName("Should handle zero value")
        void testDiscreteZero() {
            try (var action = DISCRETE.get(0)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle negative value")
        void testDiscreteNegative() {
            try (var action = DISCRETE.get(-5)) {
                assertTrue(action.isValid());
            }
        }
    }

    @Nested
    @DisplayName("BOX Action Space Tests")
    class BoxTests {

        @Test
        @DisplayName("Should create ActionResult from single double")
        void testBoxFromDouble() {
            try (var action = BOX.get(0.5)) {
                assertNotNull(action);
                assertEquals(BOX, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should create ActionResult from double array")
        void testBoxFromDoubleArray() {
            double[] values = {0.5, -0.3, 1.0, 0.0};
            try (var action = BOX.get(values)) {
                assertNotNull(action);
                assertEquals(BOX, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should create ActionResult from float array")
        void testBoxFromFloatArray() {
            float[] values = {0.5f, -0.3f, 1.0f};
            try (var action = BOX.get(values)) {
                assertNotNull(action);
                assertEquals(BOX, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle empty double array")
        void testBoxEmptyArray() {
            try (var action = BOX.get(new double[]{})) {
                assertNotNull(action);
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle single element array")
        void testBoxSingleElementArray() {
            try (var action = BOX.get(new double[]{1.5})) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle extreme values")
        void testBoxExtremeValues() {
            double[] values = {Double.MAX_VALUE, Double.MIN_VALUE, 0.0, -1.0};
            try (var action = BOX.get(values)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should throw exception for unsupported int")
        void testBoxDoesNotSupportInt() {
            assertThrows(UnsupportedOperationException.class, () -> BOX.get(5));
        }
    }

    @Nested
    @DisplayName("MULTI_DISCRETE Action Space Tests")
    class MultiDiscreteTests {

        @Test
        @DisplayName("Should create ActionResult from int array")
        void testMultiDiscreteFromIntArray() {
            int[] values = {2, 1, 3, 0};
            try (var action = MULTI_DISCRETE.get(values)) {
                assertNotNull(action);
                assertEquals(MULTI_DISCRETE, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should create ActionResult from long array")
        void testMultiDiscreteFromLongArray() {
            long[] values = {100L, 200L, 300L};
            try (var action = MULTI_DISCRETE.get(values)) {
                assertNotNull(action);
                assertEquals(MULTI_DISCRETE, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle empty array")
        void testMultiDiscreteEmptyArray() {
            try (var action = MULTI_DISCRETE.get(new int[]{})) {
                assertNotNull(action);
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle negative values")
        void testMultiDiscreteNegativeValues() {
            int[] values = {-1, 0, 1, -5};
            try (var action = MULTI_DISCRETE.get(values)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should throw exception for unsupported single value")
        void testMultiDiscreteDoesNotSupportSingleValue() {
            assertThrows(UnsupportedOperationException.class, () -> MULTI_DISCRETE.get(5));
        }

        @Test
        @DisplayName("Should throw exception for unsupported double array")
        void testMultiDiscreteDoesNotSupportDoubleArray() {
            assertThrows(UnsupportedOperationException.class,
                    () -> MULTI_DISCRETE.get(new double[]{1.0, 2.0}));
        }
    }

    @Nested
    @DisplayName("MULTI_BINARY Action Space Tests")
    class MultiBinaryTests {

        @Test
        @DisplayName("Should create ActionResult from boolean array")
        void testMultiBinaryFromBooleanArray() {
            boolean[] values = {true, false, true, true, false};
            try (var action = MULTI_BINARY.get(values)) {
                assertNotNull(action);
                assertEquals(MULTI_BINARY, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should create ActionResult from int array (binary conversion)")
        void testMultiBinaryFromIntArray() {
            int[] values = {1, 0, 1, 0};
            try (var action = MULTI_BINARY.get(values)) {
                assertNotNull(action);
                assertEquals(MULTI_BINARY, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle empty boolean array")
        void testMultiBinaryEmptyArray() {
            try (var action = MULTI_BINARY.get(new boolean[]{})) {
                assertNotNull(action);
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle all true values")
        void testMultiBinaryAllTrue() {
            boolean[] values = {true, true, true};
            try (var action = MULTI_BINARY.get(values)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle all false values")
        void testMultiBinaryAllFalse() {
            boolean[] values = {false, false, false};
            try (var action = MULTI_BINARY.get(values)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should throw exception for unsupported single value")
        void testMultiBinaryDoesNotSupportSingleValue() {
            assertThrows(UnsupportedOperationException.class, () -> MULTI_BINARY.get(1));
        }
    }

    @Nested
    @DisplayName("TEXT Action Space Tests")
    class TextTests {

        @Test
        @DisplayName("Should create ActionResult from string")
        void testTextFromString() {
            try (var action = TEXT.get("Hello, World!")) {
                assertNotNull(action);
                assertEquals(TEXT, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle empty string")
        void testTextEmptyString() {
            try (var action = TEXT.get("")) {
                assertNotNull(action);
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle unicode characters")
        void testTextUnicode() {
            try (var action = TEXT.get("Hello 世界 🌍")) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle long string")
        void testTextLongString() {
            String longText = "a".repeat(10000);
            try (var action = TEXT.get(longText)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should throw exception for unsupported int")
        void testTextDoesNotSupportInt() {
            assertThrows(UnsupportedOperationException.class, () -> TEXT.get(5));
        }

        @Test
        @DisplayName("Should throw exception for unsupported array")
        void testTextDoesNotSupportArray() {
            assertThrows(UnsupportedOperationException.class, () -> TEXT.get(new int[]{1, 2}));
        }
    }

    @Nested
    @DisplayName("UNKNOWN Action Space Tests")
    class UnknownTests {

        @Test
        @DisplayName("Should throw exception for convert")
        void testUnknownConvert() {
            PyObject pyObj = pyLong(5);
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.convert(pyObj));
            decRef(pyObj);
        }

        @Test
        @DisplayName("Should throw exception for all get methods")
        void testUnknownDoesNotSupportAnyMethod() {
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.get(5));
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.get(5L));
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.get(5.0));
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.get("test"));
            assertThrows(UnsupportedOperationException.class, () -> UNKNOWN.get(new int[]{}));
        }
    }

    @Nested
    @DisplayName("detectActionSpaceType Tests")
    class DetectActionSpaceTypeTests {

        @Test
        @DisplayName("Should detect Discrete space")
        void testDetectDiscrete() {
            exec("import gymnasium as gym; space = gym.spaces.Discrete(5)");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(DISCRETE, detected);
            decRef(pySpace);
            exec("del space");
        }

        @Test
        @DisplayName("Should detect Box space")
        void testDetectBox() {
            exec("import gymnasium as gym; import numpy as np; space = gym.spaces.Box(low=0, high=1, shape=(3,))");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(BOX, detected);
            decRef(pySpace);
            exec("del space");
        }

        @Test
        @DisplayName("Should detect MultiDiscrete space")
        void testDetectMultiDiscrete() {
            exec("import gymnasium as gym; import numpy as np; space = gym.spaces.MultiDiscrete([3, 4, 5])");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(MULTI_DISCRETE, detected);
            decRef(pySpace);
            exec("del space");
        }

        @Test
        @DisplayName("Should detect MultiBinary space")
        void testDetectMultiBinary() {
            exec("import gymnasium as gym; space = gym.spaces.MultiBinary(5)");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(MULTI_BINARY, detected);
            decRef(pySpace);
            exec("del space");
        }

        @Test
        @DisplayName("Should detect Text space")
        void testDetectText() {
            exec("import gymnasium as gym; space = gym.spaces.Text(max_length=100)");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(TEXT, detected);
            decRef(pySpace);
            exec("del space");
        }

        @Test
        @DisplayName("Should return UNKNOWN for unknown space")
        void testDetectUnknown() {
            exec("class CustomSpace: pass");
            exec("space = CustomSpace()");
            PyObject pySpace = eval("space");

            ActionSpaceType detected = ActionSpaceType.detectActionSpaceType(pySpace);

            assertEquals(UNKNOWN, detected);
            decRef(pySpace);
            exec("del space; del CustomSpace");
        }
    }

    @Nested
    @DisplayName("ActionResult Tests")
    class ActionResultTests {

        @Test
        @DisplayName("Should close ActionResult properly")
        void testActionResultClose() {
            var action = DISCRETE.get(5);

            assertFalse(action.isClosed());
            assertTrue(action.isValid());

            action.close();

            assertTrue(action.isClosed());
            assertFalse(action.isValid());
        }

        @Test
        @DisplayName("Should throw exception when closing twice")
        void testActionResultDoubleClose() {
            var action = DISCRETE.get(5);
            action.close();

            assertThrows(IllegalStateException.class, action::close);
        }

        @Test
        @DisplayName("Should work with try-with-resources")
        void testActionResultTryWithResources() {
            assertDoesNotThrow(() -> {
                try (var action = DISCRETE.get(5)) {
                    assertTrue(action.isValid());
                }
            });
        }

        @Test
        @DisplayName("Should return correct space type")
        void testActionResultSpaceType() {
            try (var action = BOX.get(1.5)) {
                assertEquals(BOX, action.spaceType());
            }
        }

        @Test
        @DisplayName("Should handle convert method")
        void testActionResultConvert() {
            PyObject pyObj = pyLong(42);

            try (var action = DISCRETE.convert(pyObj)) {
                assertNotNull(action);
                assertEquals(DISCRETE, action.spaceType());
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should become invalid after close")
        void testActionResultInvalidAfterClose() {
            var action = DISCRETE.get(10);
            assertTrue(action.isValid());

            action.close();

            assertFalse(action.isValid());
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCasesTests {

        @Test
        @DisplayName("Should handle large arrays")
        void testLargeArrays() {
            int[] largeArray = new int[10000];
            for (int i = 0; i < largeArray.length; i++) {
                largeArray[i] = i;
            }

            try (var action = MULTI_DISCRETE.get(largeArray)) {
                assertTrue(action.isValid());
            }
        }

        @Test
        @DisplayName("Should handle special double values")
        void testSpecialDoubleValues() {
            double[] specialValues = {
                    Double.POSITIVE_INFINITY,
                    Double.NEGATIVE_INFINITY,
                    Double.NaN,
                    0.0,
                    -0.0
            };

            try (var action = BOX.get(specialValues)) {
                assertTrue(action.isValid());
            }
        }

        @ParameterizedTest
        @EnumSource(value = ActionSpaceType.class, names = {"DISCRETE", "BOX", "MULTI_DISCRETE", "MULTI_BINARY", "TEXT"})
        @DisplayName("Should have unique behavior for each space type")
        void testEachSpaceTypeIsUnique(ActionSpaceType spaceType) {
            assertNotNull(spaceType);
            assertNotEquals(UNKNOWN, spaceType);
        }

        @Test
        @DisplayName("Should handle rapid creation and closure")
        void testRapidCreationAndClosure() {
            assertDoesNotThrow(() -> {
                for (int i = 0; i < 1000; i++) {
                    try (var action = DISCRETE.get(i)) {
                        assertTrue(action.isValid());
                    }
                }
            });
        }

        @Test
        @DisplayName("Should maintain integrity across multiple actions")
        void testMultipleActionsSimultaneously() {
            try (var action1 = DISCRETE.get(1);
                 var action2 = BOX.get(2.5);
                 var action3 = TEXT.get("test")) {

                assertTrue(action1.isValid());
                assertTrue(action2.isValid());
                assertTrue(action3.isValid());

                assertEquals(DISCRETE, action1.spaceType());
                assertEquals(BOX, action2.spaceType());
                assertEquals(TEXT, action3.spaceType());
            }
        }
    }

    @Nested
    @DisplayName("Memory and Reference Counting Tests")
    class MemoryTests {

        @Test
        @DisplayName("Should not leak memory on repeated operations")
        void testNoMemoryLeak() {
            assertDoesNotThrow(() -> {
                for (int i = 0; i < 10000; i++) {
                    try (var action = DISCRETE.get(i)) {
                        // Action is automatically closed
                    }
                }
            });
        }

        @Test
        @DisplayName("Should handle nested try-with-resources")
        void testNestedTryWithResources() {
            assertDoesNotThrow(() -> {
                try (var action1 = DISCRETE.get(1)) {
                    try (var action2 = BOX.get(2.0)) {
                        assertTrue(action1.isValid());
                        assertTrue(action2.isValid());
                    }
                    assertTrue(action1.isValid());
                }
            });
        }
    }
}
