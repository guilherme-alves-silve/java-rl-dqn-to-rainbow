package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection;
import br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer;
import lombok.Cleanup;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

import static br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection.*;
import static br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer.noisyLayer;
import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.*;

/**
 * Categorical (C51) distributional Q-network.
 *
 * <p>Reference: <a href="https://arxiv.org/abs/1707.06887">A Distributional Perspective on
 * Reinforcement Learning</a>.
 *
 * <p>The network outputs a softmax distribution over {@code atoms} support values for each
 * action. The shape returned by {@link #forwardDist(NDList)} is
 * {@code (batch, atoms, actions)}; the Q-value for each (s, a) is recovered as
 * {@code sum(z_i * p_i(s, a))} over the atom dimension (handled by {@link #forward(NDArray)}).
 *
 * <p>Default support: {@code Vmin = -10}, {@code Vmax = 10}, {@code atoms = 51}.
 */
@Slf4j
public class CategoricalQNoisyNetworkMLP implements INoisyNetwork, ICategoricalNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final int atoms;
    private final List<NoisyLayer> noisyLayers;
    private final CategoricalBellmanProjection catProj;
    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;
    private final NDArray support;

    public CategoricalQNoisyNetworkMLP(int observations,
                                       int actions,
                                       NDManager parent) {
        this(observations, actions, N_ATOMS, V_MIN, V_MAX, parent);
    }

    public CategoricalQNoisyNetworkMLP(int observations,
                                       int actions,
                                       int atoms,
                                       float vMin,
                                       float vMax,
                                       NDManager parent) {
        var catProj = new CategoricalBellmanProjection(atoms, vMin, vMax);
        this(observations, actions, catProj, null, null, parent);
    }

    private CategoricalQNoisyNetworkMLP(int observations, int actions, CategoricalBellmanProjection catProj, NDManager parent) {
        this(observations, actions, catProj, null, null, parent);
    }

    @SneakyThrows
    public CategoricalQNoisyNetworkMLP(int observations,
                                       int actions,
                                       CategoricalBellmanProjection catProj,
                                       Path modelPath,
                                       String prefix,
                                       NDManager parent) {
        this.observations = observations;
        this.actions = actions;
        this.catProj = catProj;
        this.subManager = subMgr(parent, getClass());

        // Configure C51 Parameters
        this.atoms = catProj.atoms();
        this.support = catProj.support(subManager);
        this.model = newModel(getClass(), subManager.getDevice());
        this.noisyLayers = new ArrayList<>();
        this.net = new SequentialBlock();
        net.add(linear(128))
           .add(Activation::relu)
           .add(addAndGet(noisyLayers, noisyLayer(128)))
           .add(Activation::relu)
           .add(addAndGet(noisyLayers, noisyLayer(actions * atoms)));
        model.setBlock(net);

        this.parameterStore = new ParameterStore(subManager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(subManager, DataType.FLOAT32, new Shape(1, observations));
            initGradients(model.getBlock());
            this.training = true;
        }
    }

    @Override
    public NDArray forwardDist(NDList inputs, final UnaryOperator<NDArray> block) {
        @Cleanup var logits = safeForwardSingle(subManager, net, parameterStore, inputs, training).singletonOrThrow();
        return scoped(it -> {
            var probDist = it.reshape(N_BATCH, actions, atoms)
                             .softmax(LAST_AXIS);
            var out = block.apply(probDist);
            out.tempAttach(it.getManager());
            return out;
        }, logits);
    }

    /**
     * Forward pass returning raw logits of shape {@code (batch, actions, atoms)}.
     *
     * @param input the network input tensor
     * @param block transformation applied after reshaping (e.g., {@code nd -> nd.logSoftmax(LAST_AXIS)})
     * @return logits tensor of shape {@code (batch, actions, atoms)}
     */
    @Override
    public NDArray forwardLogits(NDArray input, final UnaryOperator<NDArray> block) {
        @Cleanup var logits = safeForwardSingle(subManager, net, parameterStore, new NDList(input), training).singletonOrThrow();
        return scoped(it -> {
            var logitsDist = it.reshape(N_BATCH, actions, atoms);
            var out = block.apply(logitsDist);
            out.tempAttach(it.getManager());
            return out;
        }, logits);
    }

    /**
     * Computes the Q-value for each (batch, action) pair as the expectation of the
     * categorical return distribution over the atom support {@code z}:
     * {@code Q(s, a) = sum_i z_i * p_i(s, a)}. Works for both online ({@code p(s, a)})
     * and target-network ({@code p(s', a)}) distributions.
     *
     * @param distribution categorical distribution over atoms, shape {@code (batch, actions, atoms)}
     * @return Q-values, shape {@code (batch, actions)}
     * @throws IllegalStateException if {@code distribution} is not rank 3
     */
    @Override
    public NDArray qValuesFromDist(NDArray distribution) {
        if (distribution.getShape().dimension() != 3) {
            throw new IllegalStateException("Invalid shape, must be (batch, actions, atoms)!");
        }
        // (batch, actions, atoms) * (1, atoms) -> (batch, actions)
        return scoped(array -> {
            var dist = array[0];
            var sup = array[1];
            return dist.mul(sup).sum(LAST_AXIS_ARR);
        }, distribution, support);
    }

    @Override
    public NDArray projectBellman(final NDArray probNextDist,
                                  final NDArray rewards,
                                  final NDArray dones,
                                  final float gamma) {
        return catProj.project(probNextDist, rewards, dones, gamma);
    }

    public int atoms() {
        return atoms;
    }

    /**
     * Return an array of ones so we
     * broadcast the actions to transform from (batch, 1, 1)
     * to (batch, 1, atoms), e.g.: (32, 1, 1) -> (32, 1, 51)
     * @return NDArray of shape (1, 1, atoms), e.g.: (1, 1, 51)
     */
    public NDArray newAtomsBroadcaster(NDManager external) {
        return external.ones(new Shape(1, 1, atoms))
                .stopGradient();
    }

    @Override
    public NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    @Override
    @SneakyThrows
    public void save(Path modelPath, String newModelName) {
        this.model.save(modelPath, newModelName);
    }

    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    @Override
    public void eval() {
        this.training = false;
    }

    @Override
    public void train() {
        this.training = true;
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public IDeepQNetwork clone() {
        var cloned = new CategoricalQNoisyNetworkMLP(observations, actions,
                                                catProj, subManager.getParentManager());
        setName(cloned.subManager, "clone");
        copy(model.getBlock(), cloned.model.getBlock());
        return cloned;
    }

    @Override
    public NDManager subManager() {
        return subManager;
    }

    @Override
    public void close() {
        subManager.close();
        model.close();
    }

    @Override
    public void resetNoise() {
        this.noisyLayers.forEach(NoisyLayer::resetNoise);
    }
}
