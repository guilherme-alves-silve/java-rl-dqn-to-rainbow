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
import lombok.Cleanup;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.util.function.UnaryOperator;

import static br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection.*;
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
public class CategoricalQNetworkMLP implements IDeepQNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final int atoms;
    private final CategoricalBellmanProjection catProj;
    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;
    private final NDArray support;

    public CategoricalQNetworkMLP(int observations,
                                  int actions,
                                  NDManager parent) {
        this(observations, actions, N_ATOMS, V_MIN, V_MAX, parent);
    }

    public CategoricalQNetworkMLP(int observations,
                                  int actions,
                                  int atoms,
                                  float vMin,
                                  float vMax,
                                  NDManager parent) {
        var catProj = new CategoricalBellmanProjection(atoms, vMin, vMax);
        this(observations, actions, catProj, null, null, parent);
    }

    private CategoricalQNetworkMLP(int observations, int actions, CategoricalBellmanProjection catProj, NDManager parent) {
        this(observations, actions, catProj, null, null, parent);
    }

    @SneakyThrows
    CategoricalQNetworkMLP(int observations,
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
        this.net = new SequentialBlock();
        net.add(linear(128))
           .add(Activation::relu)
           .add(linear(128))
           .add(Activation::relu)
           .add(linear((long) actions * atoms));
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

    /**
     * Applies softmax transformation to action distributions over atom supports.
     *
     * <p>For each action, the distribution over atoms is converted to probabilities
     * using softmax.
     *
     * <p>The output shape is {@code (batch, actions, atoms)}, with softmax applied along the atoms
     * dimension (index 2).</p>
     *
     * <p>Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/softmax-regression-djl.html">
     * Softmax Regression — DJL</a></p>
     *
     * @param inputs the input logits for each action-atom pair
     * @return the softmax transformed probabilities in the specified shape
     */
    public NDArray forwardDist(NDList inputs) {
        return forwardDist(inputs, UnaryOperator.identity());
    }

    public NDArray forwardDist(NDArray input, final UnaryOperator<NDArray> block) {
        return forwardDist(new NDList(input), block);
    }

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
     * Applies log-softmax transformation to action distributions over atom supports.
     *
     * <p>For each action, the distribution over atoms is converted to probabilities
     * using softmax. While the C51 paper describes using standard softmax for
     * {@code p(s, a; θ)}, log-softmax is preferred for numerical stability as it
     * prevents overflow during exponentiation.</p>
     *
     * <p>The output shape is {@code (batch, actions, atoms)}, with log-softmax applied along the atoms
     * dimension (index 2).</p>
     *
     * <p>Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/softmax-regression-djl.html">
     * Softmax Regression — DJL</a></p>
     *
     * @param input the input logits for each action-atom pair
     * @return the log-softmax transformed probabilities in the specified shape
     */
    public NDArray forwardLogDist(NDArray input, final UnaryOperator<NDArray> block) {
        return forwardLogDist(new NDList(input), block);
    }

    public NDArray forwardLogDist(NDList inputs, final UnaryOperator<NDArray> block) {
        @Cleanup var logits = safeForwardSingle(subManager, net, parameterStore, inputs, training).singletonOrThrow();
        return scoped(it -> {
            var probDist = it.reshape(N_BATCH, actions, atoms)
                    .logSoftmax(LAST_AXIS);
            var out = block.apply(probDist);
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

    public NDArray projectBellman(final NDArray probNextDist,
                                  final NDArray rewards,
                                  final NDArray dones,
                                  final float gamma) {
        return catProj.project(probNextDist, rewards, dones, gamma);
    }

    @Override
    public NDList forward(NDList input) {
        @Cleanup var dist = forwardDist(input);
        return new NDList(qValuesFromDist(dist));
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
    public IDeepQNetwork clone() {
        var cloned = new CategoricalQNetworkMLP(observations, actions,
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
}
