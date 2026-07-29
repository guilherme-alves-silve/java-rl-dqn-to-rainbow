package br.com.guialves.rflr.algorithms.networks.layers;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import lombok.Cleanup;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;

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
 *
 * <p>Initialisation:
 * <ul>
 *   <li>{@code mu_w, mu_b ~ U(-1/sqrt(n), 1/sqrt(n))} where {@code n} is the number of inputs</li>
 *   <li>{@code sigma_w, sigma_b = 0.5 / sqrt(n)} (constant initialization, not random)</li>
 * </ul>
 *
 * <p>The noise is re-sampled on every forward call. The agent should call
 * {@link #resetNoise()} before each optimization step (training) and before each
 * action selection (inference) to drive exploration with a fresh noise sample.
 * <p>Reference:
 *  <a href="https://arxiv.org/abs/1706.10295">Noisy Networks for Exploration</a>.
 *  <a href="https://molab.marimo.io/github/Curt-Park/rainbow-is-all-you-need/blob/master/05_noisy_net.py">Python - Noisy Net</a>
 */
public class NoisyLayer extends AbstractBlock implements AutoCloseable {

    private static final byte VERSION = 1;

    public static NoisyLayer noisyLayer(int outFeatures) {
        return new NoisyLayer(outFeatures);
    }

    private final int outFeatures;
    private final Parameter weightMu;
    private final Parameter weightSigma;
    private final Parameter biasMu;
    private final Parameter biasSigma;
    private FactorizedNoise.Noise noise;

    public NoisyLayer(int outFeatures) {
        super(VERSION);
        this.outFeatures = outFeatures;
        this.weightMu = addParameter(Parameter.builder()
                .setName("weightMu")
                .setType(Parameter.Type.WEIGHT)
                .build());
        this.weightSigma = addParameter(Parameter.builder()
                .setName("weightSigma")
                .setType(Parameter.Type.WEIGHT)
                .build());
        this.biasMu = addParameter(Parameter.builder()
                .setName("biasMu")
                .setType(Parameter.Type.BIAS)
                .build());
        this.biasSigma = addParameter(Parameter.builder()
                .setName("biasSigma")
                .setType(Parameter.Type.BIAS)
                .build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore paramStore,
                                     NDList inputs,
                                     boolean training,
                                     PairList<String, Object> params) {
        @Cleanup var sub = subMgr(inputs.getManager(), "noisy-layer");
        var device = inputs.getManager().getDevice();
        var input = inputs.singletonOrThrow();
        var wMu = paramStore.getValue(weightMu, device, training);
        var wSigma = paramStore.getValue(weightSigma, device, training);
        var bMu = paramStore.getValue(biasMu, device, training);
        var bSigma = paramStore.getValue(biasSigma, device, training);
        sub.tempAttachAll(wMu, wSigma, bMu, bSigma);

        if (training) {
            int inFeatures = (int) input.getShape().getLastDimension();
            ensureNoiseIsSampled(input.getManager(), inFeatures, outFeatures);
            // w = \mu^w + \sigma^w \odot \epsilon^w
            var w = wMu.add(wSigma.mul(noise.epsWeight()));
            // b = \mu^b + \sigma^b \odot \epsilon^b
            var b = bMu.add(bSigma.mul(noise.epsBias()));
            // y = w \cdot x + b, in ML is, x \cdot w + b
            return Linear.linear(input, w, b);
        }

        return Linear.linear(input, wMu, bMu);
    }

    /**
     * The method initialize of the AbstractBlock
     * will call first the prepare method, initialize of
     * each parameter, then initializeChildBlocks,
     * but in this case, initializeChildBlocks was not
     * necessary.
     * @param inputShapes the shapes of inputs
     */
    @Override
    public void prepare(Shape[] inputShapes) {
        int inFeatures = (int) inputShapes[0].getLastDimension();
        var weightShape = new Shape(outFeatures, inFeatures);
        var biasShape = new Shape(outFeatures);
        weightMu.setShape(weightShape);
        weightSigma.setShape(weightShape);
        biasMu.setShape(biasShape);
        biasSigma.setShape(biasShape);

        // muW, muB ~ U(-1 / sqrt(p_in), 1 / sqrt(p_in))
        weightMu.setInitializer(NoisyLayerInit.ofMu(inFeatures));
        biasMu.setInitializer(NoisyLayerInit.ofMu(inFeatures));
        // sigmaW = 0.5 / sqrt(p_in)
        weightSigma.setInitializer(NoisyLayerInit.ofSigma(inFeatures));
        // sigmaB = 0.5 / sqrt(p_out) -> it's asymmetric
        biasSigma.setInitializer(NoisyLayerInit.ofSigma(outFeatures));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // example: (batchSize, height, width)
        long batchSize = inputShapes[0].get(0);
        return new Shape[] {new Shape(batchSize, outFeatures)};
    }

    public void resetNoise() {
        if (noise != null && !noise.isRelease()) noise.close();
        noise = null;
    }

    private void ensureNoiseIsSampled(NDManager manager, int inFeatures, int outFeatures) {
        if (noise != null) return;
        this.noise = FactorizedNoise.sampleNoiseOuter(manager, inFeatures, outFeatures);
    }

    @Override
    public void close() {
        this.resetNoise();
    }
}
