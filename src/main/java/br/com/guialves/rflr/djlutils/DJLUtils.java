package br.com.guialves.rflr.djlutils;

import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;

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
}
