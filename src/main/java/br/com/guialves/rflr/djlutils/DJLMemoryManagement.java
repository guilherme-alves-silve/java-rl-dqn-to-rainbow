package br.com.guialves.rflr.djlutils;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import lombok.Cleanup;
import lombok.SneakyThrows;

import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.UnaryOperator;

import static java.util.stream.Collectors.*;

public class DJLMemoryManagement {

    private DJLMemoryManagement() {
        throw new IllegalStateException("No DJLMemoryManagement!");
    }

    public static void close(NDArray input) {
        if (input != null) input.close();
    }

    public static void close(NDArray inputA, NDArray inputB) {
        if (inputA != null) inputA.close();
        if (inputB != null) inputB.close();
    }

    public static void close(NDArray... arrays) {
        for (var array : arrays) {
            if (array != null) array.close();
        }
    }

    /**
     * Close and set null in each element to avoid memory leaks.
     * @param arrays Each element will be closed and null set
     */
    @SneakyThrows
    public static void erase(AutoCloseable[] arrays) {
        for (int i = 0; i < arrays.length; ++i) {
            if (arrays[i] != null) arrays[i].close();
            arrays[i] = null;
        }
    }

    public static NDArray transfer(NDManager manager, NDArray oldVal, NDArray newVal) {
        oldVal.close();
        newVal.attach(manager);
        return newVal;
    }

    public static NDArray scoped(final UnaryOperator<NDArray> block,
                                 final NDArray input) {
        @Cleanup var sub = subMgr(input, "scoped-input");
        input.tempAttach(sub);
        var result = block.apply(input);
        if (result == input) {
            throw new IllegalStateException("scoped block returned the input NDArray itself");
        }

        return sub.ret(result);
    }

    public static float scopedToFloat(final UnaryOperator<NDArray> block,
                                      final NDArray input) {
        @Cleanup var out = scoped(block, input);
        return out.getFloat();
    }

    public static NDArray scoped(final BinaryOperator<NDArray> block,
                                 final NDArray inputA,
                                 final NDArray inputB) {
        if (inputA.getManager() != inputB.getManager()) {
            throw new IllegalArgumentException("scoped inputs must belong to the same NDManager");
        }

        @Cleanup var sub = subMgr(inputA, "scoped-a-b");
        inputA.tempAttach(sub);
        inputB.tempAttach(sub);

        var result = sub.ret(block.apply(inputA, inputB));
        if (result == inputA || result == inputB) {
            throw new IllegalStateException("scoped block returned any input NDArray itself");
        }
        return result;
    }

    public static NDArray scoped(final Function<NDArray[], NDArray> block,
                                 final NDArray... arrays) {
        if (arrays.length == 0) throw new IllegalArgumentException("arrays must contain elements!");

        @Cleanup var sub = subMgr(arrays[0], "scoped-arrays");
        sub.tempAttachAll(arrays);

        var result = sub.ret(block.apply(arrays));
        for (var element : arrays) {
            if (element == result) {
                throw new IllegalStateException("scoped block returned any input NDArray itself");
            }
        }

        return result;
    }

    public static NDArray scoped(final BiFunction<NDArray, NDArray[], NDArray> block,
                                 final NDArray input,
                                 final NDArray... arrays) {
        if (arrays.length == 0) throw new IllegalArgumentException("arrays must contain elements!");

        @Cleanup var sub = subMgr(arrays[0], "scoped-a-arrays");
        input.tempAttach(sub);
        sub.tempAttachAll(arrays);

        var result = sub.ret(block.apply(input, arrays));
        for (var element : arrays) {
            if (element == result) {
                throw new IllegalStateException("scoped block returned any input NDArray itself");
            }
        }

        return result;
    }

    public static NDList safeForwardSingle(NDManager manager,
                                           Block block,
                                           ParameterStore parameterStore,
                                           NDList inputs,
                                           boolean training) {
        if (inputs.size() != 1) throw new IllegalArgumentException("The dueling dqn just accepts one input!");
        @Cleanup var sub = subMgr(manager, "safe-forward-single");
        inputs.getFirst().tempAttach(sub);
        var output = block.forward(parameterStore, inputs, training);
        output.attach(manager);
        return output;
    }

    public static void debugDump(NDManager manager) {
        if (manager instanceof BaseNDManager base) {
            IO.println("Debug dump NDManager:");
            debugDump(base, 0);
        } else {
            IO.println("NDManager is not a BaseNDManager: " + manager.getClass());
        }
    }

    /**
     * This class is used to debug the resources per
     * level of each manager, but now the name is printed
     * to help find the resources that is leaking,
     * the original debugDump didn't print the name,
     * so to find the resource, is much harder.
     * @param manager NDManager to explore
     * @param level the level below the upper manager
     */
    @SneakyThrows
    @SuppressWarnings("unchecked")
    private static void debugDump(NDManager manager, int level) {
        var sb = new StringBuilder(120);
        sb.repeat("    ", Math.max(0, level));

        var uidField = BaseNDManager.class.getDeclaredField("uid");
        uidField.setAccessible(true);
        var resourcesField = BaseNDManager.class.getDeclaredField("resources");
        resourcesField.setAccessible(true);

        var uid = uidField.get(manager);
        var resources = (ConcurrentHashMap<String, AutoCloseable>) resourcesField.get(manager);

        sb.append("\\--- NDManager[").append(manager.getName())
                .append(" | uid=").append(uid)
                .append("] resources=").append(resources.size());

        // Divide by resource type
        var byType = resources.values().stream()
                .collect(groupingBy(resource -> resource == null ?
                                        "null" : resource.getClass().getSimpleName(),
                        counting()));
        sb.append(" {").append(byType.entrySet().stream()
                        .map(e -> e.getKey() + "=" + e.getValue())
                        .sorted()
                        .collect(joining(", ")))
                .append("}");
        IO.println(sb);

        for (var element : resources.values()) {
            if (element instanceof BaseNDManager innerBase) {
                debugDump(innerBase, level + 1);
            }
        }
    }

    public static int managedArrayCount(NDManager manager) {
        if (manager instanceof BaseNDManager base) {
            return base.getManagedArrays().size();
        }

        return -1;
    }

    public static NDArray setName(NDArray array, String name) {
        array.setName(name + "-" + array.getName());
        return array;
    }

    public static NDManager subMgr(NDArray array, String name) {
        return subMgr(array.getManager(), name);
    }

    public static NDManager subMgr(NDManager manager, Class<?> clazz) {
        return subMgr(manager, clazz.getSimpleName());
    }

    public static NDManager subMgr(NDManager manager, String name) {
        var sub = manager.newSubManager();
        sub.setName(name + "-" + sub.getName());
        return sub;
    }

    public static Model newModel(Class<?> clazz) {
        return newModel(clazz.getSimpleName(), null);
    }

    public static Model newModel(String name) {
        return newModel(name, null);
    }

    public static Model newModel(Class<?> clazz, Device device) {
        return newModel(clazz.getSimpleName(), device);
    }

    public static Model newModel(String name, Device device) {
        var model = Model.newInstance(name, device);
        var mgr = model.getNDManager();
        mgr.setName(name + "-" + mgr.getName());
        return model;
    }
}
