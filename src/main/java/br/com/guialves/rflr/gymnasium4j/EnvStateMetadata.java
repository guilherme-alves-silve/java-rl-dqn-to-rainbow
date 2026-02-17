package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.utils.NumPy2DJLTypeMapper;
import lombok.Getter;
import lombok.experimental.Accessors;
import org.bytedeco.cpython.PyObject;

import java.util.Arrays;

import static br.com.guialves.rflr.python.PythonDataStructures.*;
import static br.com.guialves.rflr.python.PythonRuntime.*;

@Accessors(fluent = true)
public class EnvStateMetadata {

    protected final int[] shape;
    @Getter
    protected final String dtype;
    @Getter
    protected final Shape djlShape;
    @Getter
    protected final DataType djlType;
    @Getter
    protected final int size;

    static EnvStateMetadata fromNumpy(PyObject arr) {
        int[] shape = toIntArray(attr(arr, "shape"));
        String dtype = attrStr(arr, "dtype");

        long[] longShape = Arrays.stream(shape).mapToLong(i -> i).toArray();
        DataType djlType = NumPy2DJLTypeMapper.numpyToDjl(dtype);

        int elements = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        int size = elements * NumPy2DJLTypeMapper.bytesPerElement(dtype);

        return new EnvStateMetadata(shape, dtype,
                new Shape(longShape), djlType, size);
    }

    EnvStateMetadata(
            int[] shape,
            String dtype,
            Shape djlShape,
            DataType djlType,
            int size
    ) {
        this.shape = shape;
        this.dtype = dtype;
        this.djlShape = djlShape;
        this.djlType = djlType;
        this.size = size;
    }
}
