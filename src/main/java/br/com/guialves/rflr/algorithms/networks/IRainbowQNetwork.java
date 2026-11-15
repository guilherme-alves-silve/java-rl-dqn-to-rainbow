package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import br.com.guialves.rflr.algorithms.networks.distributional.CategoricalBellmanProjection;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingCategoricalLayer;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingLayer;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.algorithms.networks.layers.NoisyLayer;
import lombok.Cleanup;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.experimental.Accessors;
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
public interface IRainbowQNetwork extends INoisyNetwork, ICategoricalNetwork {

    int atoms();

    /**
     * Return an array of ones so we
     * broadcast the actions to transform from (batch, 1, 1)
     * to (batch, 1, atoms), e.g.: (32, 1, 1) -> (32, 1, 51)
     * @return NDArray of shape (1, 1, atoms), e.g.: (1, 1, 51)
     */
    NDArray newAtomsBroadcaster(NDManager external);
}
