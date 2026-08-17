package br.com.guialves.rflr.algorithms.networks.layers.activation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Randomized Leaky Rectified Linear Unit (RReLU).
 *
 * <p>During training, the negative slope {@code a} is sampled independently for every
 * element of the input tensor from a uniform distribution {@code U(lower, upper)}. During
 * inference, the slope is fixed to the distribution's mean, {@code (lower + upper) / 2},
 * making the layer deterministic at test time — analogous to how Dropout behaves differently
 * in training versus inference.
 *
 * <pre>
 *  RReLU(x) = x,       if x &gt;= 0
 *  RReLU(x) = a * x,   otherwise, where a ~ U(lower, upper) during training
 * </pre>
 *
 * @see <a href="https://arxiv.org/abs/1505.00853">Xu, B., Wang, N., Chen, T., &amp; Li, M.
 *     (2015). Empirical Evaluation of Rectified Activations in Convolutional Network.
 *     arXiv:1505.00853</a>
 */
public class RReLU extends AbstractBlock {

    private static final byte VERSION = 1;

    /** Default lower bound {@code l = 1/8} used in the original RReLU paper. */
    public static final float LOWER = 1f / 8f;

    /** Default upper bound {@code u = 1/3} used in the original RReLU paper. */
    public static final float UPPER = 1f / 3f;

    /** Lower bound of the uniform distribution used to sample the negative slope. */
    private final float lower;

    /** Upper bound of the uniform distribution used to sample the negative slope. */
    private final float upper;

    public static RReLU rrelu() {
        return new RReLU();
    }

    /**
     * Constructs an {@code RReLU} block using the default bounds from the original paper,
     * {@code lower = 1/8} and {@code upper = 1/3}.
     *
     * @see #RReLU(float, float)
     */
    public RReLU() {
        this(LOWER, UPPER);
    }

    /**
     * Constructs an {@code RReLU} block.
     *
     * @param lower the lower bound {@code l} of {@code U(l, u)}; must satisfy {@code 0 <= lower < upper < 1}
     * @param upper the upper bound {@code u} of {@code U(l, u)}; must satisfy {@code 0 <= lower < upper < 1}
     * @see #RReLU()
     */
    public RReLU(float lower, float upper) {
        super(VERSION);
        if (lower < 0 || upper >= 1 || lower >= upper) {
            throw new IllegalArgumentException(
                    "Expected 0 <= lower < upper < 1, got lower=" + lower + ", upper=" + upper);
        }

        this.lower = lower;
        this.upper = upper;
    }

    /**
     * Applies RReLU element-wise to the input.
     *
     * <p>In training mode, a slope tensor of the same shape as the input is sampled fresh
     * per forward pass. In inference mode, a constant slope of {@code (lower + upper) / 2}
     * is applied instead.
     *
     * @param parameterStore the parameter store, unused since this block has no learnable parameters
     * @param inputs a singleton {@link NDList} containing the input {@link NDArray}
     * @param training {@code true} if running in training mode, {@code false} for inference
     * @param params additional parameters, unused
     * @return a singleton {@link NDList} containing the activated output, same shape as the input
     */
    @Override
    protected NDList forwardInternal(ParameterStore parameterStore,
                                     NDList inputs,
                                     boolean training,
                                     PairList<String, Object> params) {
        var x = inputs.singletonOrThrow();
        var manager = x.getManager();
        var mask = x.gte(0).toType(DataType.FLOAT32, false);
        var positive = x.mul(mask);

        NDArray slope;
        if (training) {
            slope = manager.randomUniform(lower, upper, x.getShape());
        } else {
            slope = manager.full(x.getShape(), (lower + upper) / 2f);
        }

        var negative = x.mul(slope).mul(mask.sub(1).neg());
        return new NDList(positive.add(negative));
    }

    /**
     * Returns the output shapes, which are identical to the input shapes since RReLU is
     * an element-wise operation.
     *
     * @param inputShapes the input shapes
     * @return the same shapes, unchanged
     */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return inputShapes;
    }
}
