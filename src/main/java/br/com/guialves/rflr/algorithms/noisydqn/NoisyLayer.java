package br.com.guialves.rflr.algorithms.noisydqn;

/**
 * Implements a fully-connected layer with factorized Gaussian noise
 * (NoisyNet), as proposed in Fortunato et al. (2017),
 * <i>Noisy Networks for Exploration</i>.
 *
 * <h2>Noise transform</h2>
 * <p>Each raw noise component is transformed by:</p>
 * <pre>{@code
 * f(x) = sign(x) * sqrt(|x|)
 * }</pre>
 *
 * <h2>Factorization</h2>
 * <p>Given input noise {@code epsIn} and output noise {@code epsOut},
 * both sampled from N(0, 1):</p>
 * <pre>{@code
 * epsWeight = f(epsOut) . f(epsIn)^T   // outer product -> matrix [out x in]
 * epsBias = f(epsOut)                // vector [out]
 * }</pre>
 *
 * <h2>Noisy parameters</h2>
 * <pre>{@code
 * w = muW + sigmaW (*) epsWeight   // (*) = element-wise (Hadamard) product
 * b = muB + sigmaB (*) epsBias
 * y = w . x + b               // matrix-vector product
 * }</pre>
 *
 * <h2>Training</h2>
 * <p>Noise is resampled at each step and propagated through the layer:</p>
 * <pre>{@code
 * y = (muW + sigmaW (*) epsWeight) . x + (muB + sigmaB (*) epsBias)
 * }</pre>
 *
 * <h2>Inference</h2>
 * <p>Noise is dropped; only the mean parameters are used
 * (deterministic/greedy behavior):</p>
 * <pre>{@code
 * y = muW . x + muB
 * }</pre>
 */
public class NoisyLayer {


}
