package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;
import lombok.Getter;

/**
 * Initializes the mean ({@code mu}) and standard deviation ({@code sigma})
 * parameters of a {@link NoisyLayer}, following the initialization scheme
 * from Fortunato et al. (2017), <i>Noisy Networks for Exploration</i>.
 *
 * <pre>{@code
 * muW, muB  ~ U(-1 / sqrt(p_in), 1 / sqrt(p_in))
 * sigmaW    = 0.5 / sqrt(p_in)
 * sigmaB    = 0.5 / sqrt(p_out)
 * }</pre>
 *
 * <p>where:</p>
 * <ul>
 *   <li>{@code p_in}  - number of input features</li>
 *   <li>{@code p_out} - number of output features</li>
 * </ul>
 *
 * <p>Note that both {@code muW} and {@code muB} are sampled using
 * {@code p_in}, while {@code sigmaW} and {@code sigmaB} use different
 * denominators ({@code p_in} and {@code p_out} respectively) — this
 * asymmetry comes directly from the original implementation.</p>
 *
 * <p>See also:
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/parameters.html#custom-initialization">
 * DJL Custom Initialization</a></p>
 */
public class NoisyLayerInit implements Initializer {

    private static final float SIGMA_INIT = 0.5f;

    public enum InitType {
        MU, SIGMA
    }

    public static NoisyLayerInit ofMu(int size) {
        return new NoisyLayerInit(size, InitType.MU);
    }

    public static NoisyLayerInit ofSigma(int size) {
        return new NoisyLayerInit(size, InitType.SIGMA);
    }

    @Getter
    private final int size;
    @Getter
    private final InitType type;

    public NoisyLayerInit(int size, InitType type) {
        this.size = size;
        this.type = type;
    }

    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        float p = (float) Math.sqrt(size);
        return switch (type) {
            case MU -> {
                float high = 1f/p;
                float low = -high;
                yield manager.randomUniform(low, high, shape, dataType);
            }
            case SIGMA -> manager.full(shape, SIGMA_INIT/p, dataType);
        };
    }
}
