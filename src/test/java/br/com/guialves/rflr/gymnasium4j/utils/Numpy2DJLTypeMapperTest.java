package br.com.guialves.rflr.gymnasium4j.utils;

import ai.djl.ndarray.types.DataType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

class Numpy2DJLTypeMapperTest {

    @Test
    void shouldNotAllowInstantiation() {
        assertThrows(IllegalArgumentException.class, () -> {
            try {
                var constructor = Numpy2DJLTypeMapper.class.getDeclaredConstructor();
                constructor.setAccessible(true);
                constructor.newInstance();
            } catch (Exception ex) {
                throw ex.getCause();
            }
        });
    }

    @Nested
    @DisplayName("bytesPerElement Tests")
    class BytesPerElementTests {

        @ParameterizedTest
        @ValueSource(strings = {"bool", "bool_", "uint8", "int8", "byte"})
        void shouldReturn1ByteForSingleByteTypes(String dtype) {
            assertEquals(1, Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"uint16", "int16", "float16", "half"})
        void shouldReturn2BytesForTwoByteTypes(String dtype) {
            assertEquals(2, Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"uint32", "int32", "float32", "float", "single"})
        void shouldReturn4BytesForFourByteTypes(String dtype) {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"uint64", "int64", "float64", "double"})
        void shouldReturn8BytesForEightByteTypes(String dtype) {
            assertEquals(8, Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"<float32", ">float64", "=int32", "|uint8"})
        void shouldHandleEndianness(String dtype) {
            assertDoesNotThrow(() -> Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @ParameterizedTest
        @CsvSource({
                "<float32, 4",
                ">float64, 8",
                "=int32, 4",
                "|uint8, 1",
                "<int16, 2",
                ">uint64, 8"
        })
        void shouldReturnCorrectBytesWithEndianness(String dtype, int expectedBytes) {
            assertEquals(expectedBytes, Numpy2DJLTypeMapper.bytesPerElement(dtype));
        }

        @Test
        void shouldHandleUpperCase() {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement("FLOAT32"));
            assertEquals(8, Numpy2DJLTypeMapper.bytesPerElement("FLOAT64"));
            assertEquals(1, Numpy2DJLTypeMapper.bytesPerElement("BOOL"));
        }

        @Test
        void shouldHandleMixedCase() {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement("Float32"));
            assertEquals(8, Numpy2DJLTypeMapper.bytesPerElement("Double"));
            assertEquals(2, Numpy2DJLTypeMapper.bytesPerElement("Int16"));
        }

        @Test
        void shouldHandleWhitespace() {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement("  float32  "));
            assertEquals(8, Numpy2DJLTypeMapper.bytesPerElement(" float64 "));
        }

        @Test
        void shouldThrowForUnsupportedType() {
            var exception = assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.bytesPerElement("complex128"));
            assertTrue(exception.getMessage().contains("Unsupported NumPy dtype"));
        }

        @Test
        void shouldThrowForNull() {
            assertThrows(NullPointerException.class,
                    () -> Numpy2DJLTypeMapper.bytesPerElement(null));
        }

        @Test
        void shouldThrowForEmptyString() {
            assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.bytesPerElement(""));
        }

        @Test
        void shouldThrowForInvalidType() {
            assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.bytesPerElement("invalid"));
        }
    }

    @Nested
    @DisplayName("numpyToDjl Tests")
    class NumpyToDjlTests {

        @ParameterizedTest
        @ValueSource(strings = {"bool", "bool_"})
        void shouldMapBooleanTypes(String dtype) {
            assertEquals(DataType.BOOLEAN, Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"int8", "byte"})
        void shouldMapInt8Types(String dtype) {
            assertEquals(DataType.INT8, Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @Test
        void shouldMapUInt8() {
            assertEquals(DataType.UINT8, Numpy2DJLTypeMapper.numpyToDjl("uint8"));
        }

        @Test
        void shouldMapInt16() {
            assertEquals(DataType.INT16, Numpy2DJLTypeMapper.numpyToDjl("int16"));
        }

        @Test
        void shouldMapUInt16() {
            assertEquals(DataType.UINT16, Numpy2DJLTypeMapper.numpyToDjl("uint16"));
        }

        @Test
        void shouldMapInt32() {
            assertEquals(DataType.INT32, Numpy2DJLTypeMapper.numpyToDjl("int32"));
        }

        @Test
        void shouldMapUInt32() {
            assertEquals(DataType.UINT32, Numpy2DJLTypeMapper.numpyToDjl("uint32"));
        }

        @Test
        void shouldMapInt64() {
            assertEquals(DataType.INT64, Numpy2DJLTypeMapper.numpyToDjl("int64"));
        }

        @Test
        void shouldMapUInt64() {
            assertEquals(DataType.UINT64, Numpy2DJLTypeMapper.numpyToDjl("uint64"));
        }

        @ParameterizedTest
        @ValueSource(strings = {"float16", "half"})
        void shouldMapFloat16Types(String dtype) {
            assertEquals(DataType.FLOAT16, Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"float32", "float", "single"})
        void shouldMapFloat32Types(String dtype) {
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @ParameterizedTest
        @ValueSource(strings = {"float64", "double"})
        void shouldMapFloat64Types(String dtype) {
            assertEquals(DataType.FLOAT64, Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @ParameterizedTest
        @CsvSource({
                "<float32, FLOAT32",
                ">float64, FLOAT64",
                "=int32, INT32",
                "|uint8, UINT8",
                "<bool, BOOLEAN",
                ">int16, INT16"
        })
        void shouldHandleEndianness(String dtype, String expectedType) {
            assertEquals(DataType.valueOf(expectedType), Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @Test
        void shouldHandleUpperCase() {
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("FLOAT32"));
            assertEquals(DataType.INT64, Numpy2DJLTypeMapper.numpyToDjl("INT64"));
            assertEquals(DataType.BOOLEAN, Numpy2DJLTypeMapper.numpyToDjl("BOOL"));
        }

        @Test
        void shouldHandleMixedCase() {
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("Float32"));
            assertEquals(DataType.FLOAT64, Numpy2DJLTypeMapper.numpyToDjl("Double"));
            assertEquals(DataType.INT32, Numpy2DJLTypeMapper.numpyToDjl("Int32"));
        }

        @Test
        void shouldHandleWhitespace() {
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("  float32  "));
            assertEquals(DataType.INT64, Numpy2DJLTypeMapper.numpyToDjl(" int64 "));
        }

        @Test
        void shouldThrowForUnsupportedType() {
            var exception = assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.numpyToDjl("complex64"));
            assertTrue(exception.getMessage().contains("Unsupported NumPy dtype"));
        }

        @Test
        void shouldThrowForNull() {
            assertThrows(NullPointerException.class,
                    () -> Numpy2DJLTypeMapper.numpyToDjl(null));
        }

        @Test
        void shouldThrowForEmptyString() {
            assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.numpyToDjl(""));
        }

        @Test
        void shouldThrowForInvalidType() {
            assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.numpyToDjl("invalid_type"));
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {

        @ParameterizedTest
        @CsvSource({
                "bool, 1, BOOLEAN",
                "uint8, 1, UINT8",
                "int8, 1, INT8",
                "int16, 2, INT16",
                "uint16, 2, UINT16",
                "int32, 4, INT32",
                "uint32, 4, UINT32",
                "int64, 8, INT64",
                "uint64, 8, UINT64",
                "float16, 2, FLOAT16",
                "float32, 4, FLOAT32",
                "float64, 8, FLOAT64"
        })
        void shouldMapConsistently(String dtype, int expectedBytes, String expectedType) {
            assertEquals(expectedBytes, Numpy2DJLTypeMapper.bytesPerElement(dtype));
            assertEquals(DataType.valueOf(expectedType), Numpy2DJLTypeMapper.numpyToDjl(dtype));
        }

        @Test
        void shouldHandleAllEndiannessMarkers() {
            String[] markers = {"<", ">", "=", "|"};
            String[] types = {"float32", "int32", "uint8", "float64"};

            for (String marker : markers) {
                for (String type : types) {
                    String dtype = marker + type;
                    assertDoesNotThrow(() -> Numpy2DJLTypeMapper.bytesPerElement(dtype));
                    assertDoesNotThrow(() -> Numpy2DJLTypeMapper.numpyToDjl(dtype));
                }
            }
        }

        @Test
        void shouldHandleCommonNumpyTypes() {
            // Common types from Gymnasium environments
            assertDoesNotThrow(() -> {
                Numpy2DJLTypeMapper.bytesPerElement("uint8");
                Numpy2DJLTypeMapper.numpyToDjl("uint8");
            });

            assertDoesNotThrow(() -> {
                Numpy2DJLTypeMapper.bytesPerElement("float32");
                Numpy2DJLTypeMapper.numpyToDjl("float32");
            });

            assertDoesNotThrow(() -> {
                Numpy2DJLTypeMapper.bytesPerElement("float64");
                Numpy2DJLTypeMapper.numpyToDjl("float64");
            });
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        void shouldHandleMultipleSpaces() {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement("   float32   "));
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("   float32   "));
        }

        @Test
        void shouldHandleEndiannessWithSpaces() {
            assertEquals(4, Numpy2DJLTypeMapper.bytesPerElement(" <float32 "));
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl(" >float32 "));
        }

        @Test
        void shouldRejectInvalidEndiannessPrefix() {
            // These should fail because normalize removes endianness,
            // leaving invalid type names
            assertThrows(IllegalArgumentException.class,
                    () -> Numpy2DJLTypeMapper.bytesPerElement("<<float32"));
        }

        @Test
        void shouldHandleAllAliases() {
            // Test all aliases work
            assertEquals(DataType.INT8, Numpy2DJLTypeMapper.numpyToDjl("byte"));
            assertEquals(DataType.FLOAT16, Numpy2DJLTypeMapper.numpyToDjl("half"));
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("float"));
            assertEquals(DataType.FLOAT32, Numpy2DJLTypeMapper.numpyToDjl("single"));
            assertEquals(DataType.FLOAT64, Numpy2DJLTypeMapper.numpyToDjl("double"));
        }
    }
}
