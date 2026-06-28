package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.Cleanup;

import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;

public class DJLMemoryManagement {

    private DJLMemoryManagement() {
        throw new IllegalStateException("No DJLMemoryManagement!");
    }

    public static void close(NDArray a) {
        if (a != null) a.close();
    }

    public static void close(NDArray a, NDArray b) {
        if (a != null) a.close();
        if (b != null) b.close();
    }

    public static void close(NDArray... arrays) {
        for (var array : arrays) {
            if (array != null) array.close();
        }
    }

    public static NDArray transfer(NDManager manager, NDArray oldVal, NDArray newVal) {
        oldVal.close();
        newVal.attach(manager);
        return newVal;
    }

    public static NDArray scoped(final UnaryOperator<NDArray> block,
                                 final NDArray input) {
        try (var sub = input.getManager().newSubManager()) {
            input.tempAttach(sub);
            var result = block.apply(input);
            if (result == input) {
                throw new IllegalStateException("scoped block returned the input NDArray itself");
            }
            return sub.ret(result);
        }
    }

    public static float scopedToFloat(final UnaryOperator<NDArray> block,
                                      final NDArray input) {
        @Cleanup var out = scoped(block, input);
        return out.getFloat();
    }

    public static NDArray scoped(final BinaryOperator<NDArray> block,
                                 final NDArray a,
                                 final NDArray b) {
        if (a.getManager() != b.getManager()) {
            throw new IllegalArgumentException("scoped inputs must belong to the same NDManager");
        }

        try (var sub = a.getManager().newSubManager()) {
            sub.tempAttachAll(a, b);

            var result = sub.ret(block.apply(a, b));
            if (result == a || result == b) {
                throw new IllegalStateException("scoped block returned any input NDArray itself");
            }
            return result;
        }
    }

    public static NDArray scoped(final Function<NDArray[], NDArray> block,
                                 final NDArray... arrays) {
        if (arrays.length == 0) throw new IllegalArgumentException("arrays must contain elements!");

        try (var sub = arrays[0].getManager().newSubManager()) {
            sub.tempAttachAll(arrays);

            var result = sub.ret(block.apply(arrays));
            for (var element : arrays) {
                if (element == result) {
                    throw new IllegalStateException("scoped block returned any input NDArray itself");
                }
            }
            return result;
        }
    }

    public static void debugDump(NDManager manager) {
        if (manager instanceof BaseNDManager base) {
            IO.println("Debug dump NDManager:");
            base.debugDump(0);
        } else {
            IO.println("NDManager is not a BaseNDManager: " + manager.getClass());
        }
    }

    public static int managedArrayCount(NDManager manager) {
        if (manager instanceof BaseNDManager base) {
            return base.getManagedArrays().size();
        }

        return -1;
    }
}
