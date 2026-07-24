package br.com.guialves.rflr.djlutils;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;

import java.util.function.ToDoubleFunction;
import java.util.function.ToLongFunction;

import static java.util.Arrays.stream;

public class DJLUtils {

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
}
