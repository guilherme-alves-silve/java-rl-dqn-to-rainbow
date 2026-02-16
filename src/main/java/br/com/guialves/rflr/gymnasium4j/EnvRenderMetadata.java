package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.experimental.Accessors;
import org.bytedeco.cpython.PyObject;

@Accessors(fluent = true)
public class EnvRenderMetadata extends EnvStateMetadata {

    static EnvRenderMetadata fromNumpy(PyObject arr) {
        var base = EnvStateMetadata.fromNumpy(arr);
        return new EnvRenderMetadata(
                base.shape,
                base.dtype,
                base.djlShape,
                base.djlType,
                base.size
        );
    }

    private EnvRenderMetadata(
            int[] shape,
            String dtype,
            Shape djlShape,
            DataType djlType,
            int size
    ) {
        super(shape, dtype, djlShape, djlType, size);
    }

    public int height() { return shape[0]; }
    public int width()  { return shape[1]; }
    public int channels() { return shape.length > 2 ? shape[2] : 1; }
}
