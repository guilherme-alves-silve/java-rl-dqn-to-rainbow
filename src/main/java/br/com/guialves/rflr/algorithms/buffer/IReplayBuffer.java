package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

import java.util.function.Function;

public interface IReplayBuffer<T extends IExperience> extends AutoCloseable {

    void store(T exp);

    boolean enough(int batchSize);

    /**
     * @param batchSize The number of samples used to generate the mini-batch.
     * @return VecExperience record respecting this order (s, a, r, s', done)
     */
    IVecExperience sample(int batchSize);

    /**
     * Immutable capacity, not the same as size
     * @return capacity
     */
    int capacity();

    int size();

    /**
     * Converts an array of Experiences into a batched NDArray using the provided mapper.
     *
     * <p>The mapper function must add a batch dimension using {@code expandDims(0)}
     * to preserve the batch structure. For example, if each Experience maps to
     * a {@code [3, 3]} NDArray, the mapper should return {@code [1, 3, 3]}.
     *
     * <p><b>Shape behavior:</b>
     * <ul>
     *   <li>With {@code expandDims(0)}: {@code [1, 3, 3]} → {@code [N, 3, 3]} ✓</li>
     *   <li>Without {@code expandDims(0)}: {@code [3, 3]} → {@code [N*3, 3]} ✗</li>
     * </ul>
     *
     * @param batch  the array of experiences to batch
     * @param mapper maps each experience to its NDArray representation
     * @return concatenated NDArray along axis 0
     */
    default NDArray newAttachedList(final NDManager subManager,
                                    final IExperience[] batch,
                                    final Function<IExperience, NDArray> mapper) {
        var arrays = new NDList(batch.length);
        for (var exp : batch) {
            var mapped = mapper.apply(exp);
            mapped.tempAttach(subManager);
            arrays.add(mapped.expandDims(0));
        }
        var concat = NDArrays.concat(arrays, 0);
        concat.tempAttach(subManager);
        return concat.toDevice(subManager.getDevice(), false);
    }

    boolean isOpen();
}
