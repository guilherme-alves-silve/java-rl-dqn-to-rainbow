package br.com.guialves.rflr.gymnasium4j.utils;

import ai.djl.ndarray.types.DataType;

import java.util.Locale;

import static java.util.Objects.requireNonNull;

public final class Numpy2DJLTypeMapper {

    private Numpy2DJLTypeMapper() {
        throw new IllegalArgumentException("No NumpyDTypeMapper!");
    }

    public static int bytesPerElement(String dtype) {
        return switch (normalize(dtype)) {
            case "bool", "bool_", "uint8", "int8", "byte" -> 1;
            case "uint16", "int16", "float16", "half" -> 2;
            case "uint32", "int32", "float32", "float", "single" -> 4;
            case "uint64", "int64", "float64", "double" -> 8;
            default -> throw new IllegalArgumentException("Unsupported NumPy dtype: " + dtype);
        };
    }

    public static DataType numpyToDjl(String dtype) {
        return switch (normalize(dtype)) {
            case "bool", "bool_" -> DataType.BOOLEAN;
            case "int8", "byte" -> DataType.INT8;
            case "uint8" -> DataType.UINT8;
            case "int16" -> DataType.INT16;
            case "uint16" -> DataType.UINT16;
            case "int32" -> DataType.INT32;
            case "uint32" -> DataType.UINT32;
            case "int64" -> DataType.INT64;
            case "uint64" -> DataType.UINT64;
            case "float16", "half" -> DataType.FLOAT16;
            case "float32", "float", "single" -> DataType.FLOAT32;
            case "float64", "double" -> DataType.FLOAT64;
            default -> throw new IllegalArgumentException("Unsupported NumPy dtype: " + dtype);
        };
    }

    /**
     * Strip NumPy endianness markers: < > = |
     * @param dtype numpy dtype
     * @return remove endianness markers
     */
    private static String normalize(String dtype) {
        return requireNonNull(dtype, "dtype cannot be null")
                .toLowerCase(Locale.ROOT)
                .trim()
                .replaceAll("^[<>=|]", "");
    }
}
