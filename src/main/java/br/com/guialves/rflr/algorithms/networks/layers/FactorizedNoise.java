package br.com.guialves.rflr.algorithms.networks.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;

/**
 * Factorised Gaussian noise used by NoisyNets.
 *
 * <p>Reference: <a href="https://arxiv.org/abs/1706.10295">Noisy Networks for Exploration</a>.
 *
 * <p>For a standard Gaussian sample {@code x ~ N(0, 1)}, the factorized noise is
 * {@code f(x) = sign(x) * sqrt(|x|)}. This preserves the unbiasedness of the original
 * Gaussian sample while reducing the number of random variables: a single noise tensor is
 * drawn for the input and a single one for the output, and the per-weight noise is the
 * outer product of the two.
 */
public class FactorizedNoise {

    private FactorizedNoise() {
        throw new IllegalStateException("No FactorizedNoise!");
    }

    /**
     * Package private to be used by unit test.
     */
    static NDArray sampleNoise(NDManager manager, int size) {
        var gaussian = manager.randomNormal(0f, 1f, new Shape(size), DataType.FLOAT32);
        return gaussian.sign().mul(gaussian.abs().sqrt());
    }

    public static Noise sampleNoiseOuter(NDManager manager, int inSize, int outSize) {
        @Cleanup var sub = subMgr(manager, "samp-noise-outer");
        var fepsOut = sampleNoise(sub, outSize);
        var fepsIn = sampleNoise(sub, inSize);
        // (out x 1) x (1 x in) = (out x in)
        var epsW = fepsOut.reshape(outSize, 1)
                        .matMul(fepsIn.reshape(1, inSize));
        return new Noise(
                // epsWeight = f(epsOut) * f(epsIn)^T
                sub.ret(epsW),
                // epsBias = f(epsOut)
                sub.ret(fepsOut)
        );
    }

    public record Noise(NDArray epsWeight,
                        NDArray epsBias) implements AutoCloseable {

        @Override
        public void close() {
            epsWeight.close();
            epsBias.close();
        }

        public boolean isRelease() {
            return epsWeight.isReleased() && epsBias.isReleased();
        }
    }
}
