package br.com.guialves.rflr.djlutils;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;
import java.util.function.ToLongFunction;

import static java.util.Arrays.stream;

public class DJLUtils {

    private DJLUtils() {
        throw new IllegalStateException("No DJLUtils!");
    }

    public static void copy(final Block srcBlock,
                            final Block blockDst) {
        var allIdx = new NDIndex("...");
        var paramsSrc = srcBlock.getParameters();
        var paramsDst = blockDst.getParameters();
        for (Pair<String, Parameter> pair : paramsSrc) {
            var name = pair.getKey();
            var src = pair.getValue().getArray();
            var dst = paramsDst.get(name).getArray();
            dst.set(allIdx, src);
        }
    }

    public static boolean diff(final Block srcBlock,
                               final Block dstBlock) {

        var paramsSrc = srcBlock.getParameters();
        var paramsDst = dstBlock.getParameters();

        if (paramsSrc.size() != paramsDst.size()) {
            return true;
        }

        for (int i = 0; i < paramsSrc.size(); i++) {
            var src = paramsSrc.valueAt(i).getArray();
            var dst = paramsDst.valueAt(i).getArray();

            if (!src.getShape().equals(dst.getShape())) {
                return true;
            }

            if (!dst.contentEquals(src)) {
                return true;
            }
        }

        return false;
    }

    public static void freeze(final Block block) {
        for (var params : block.getParameters()) {
            params.getValue().freeze(true);
        }
    }

    public static void setGradients(final Block block) {
        for (var params : block.getParameters()) {
            params.getValue().freeze(false);
        }
    }

    public static int gpuCount() {
        return Engine.getInstance().getGpuCount();
    }

    public static <T> NDArray djlMapToLong(NDManager subManager, T[] input, ToLongFunction<T> mapper) {
        return subManager.create(stream(input).mapToLong(mapper).toArray())
                .expandDims(1)
                .toDevice(subManager.getDevice(), false);
    }

    public static <T> NDArray djlMapToFloat32(NDManager subManager, T[] input, ToDoubleFunction<T> mapper) {
        return subManager.create(stream(input).mapToDouble(mapper).toArray())
                .toType(DataType.FLOAT32, false)
                .expandDims(1)
                .toDevice(subManager.getDevice(), false);
    }

    public static boolean equalsDims(Shape shape1, Shape shape2, int start) {
        return equalsDims(shape1, shape2, start, shape1.getShape().length);
    }

    public static boolean equalsDims(Shape shape1, Shape shape2, int start, int end) {
        if (shape1.dimension() != shape2.dimension()) {
            throw new IllegalArgumentException("Different dimensions between shape1 and shape2: %d != %d".formatted(
                    shape1.dimension(), shape2.dimension()));
        }
        return Arrays.equals(shape1.getShape(), start, end, shape2.getShape(), start, end);
    }
}
