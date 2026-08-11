package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingLayer;
import br.com.guialves.rflr.algorithms.networks.layers.DuelingType;
import br.com.guialves.rflr.djlutils.DJLUtils;
import lombok.NonNull;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;

/**
 * Architecture based on the links below:
 * <a href="https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/04_dueling.py">Dueling DQN implementation in Python</a>
 * <a href="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">Reinforcement Q-Learning</a>
 * For coding reference:
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/custom-layer.html">Custom Layers</a>
 * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-djl.html">Linear Regression</a>
 * <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp-djl.html">Multilayer Perceptrons</a>
 */
@Slf4j
public class DuelingQNetworkMLP implements IDeepQNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final Model model;
    private final DuelingLayer net;
    private final ParameterStore parameterStore;

    public static DuelingQNetworkMLP withMeanType(int observations,
                                                  int actions,
                                                  NDManager parent) {
        return new DuelingQNetworkMLP(observations, actions, null, null, parent, DuelingType.MEAN);
    }

    public static DuelingQNetworkMLP withMaxType(int observations,
                                                 int actions,
                                                 NDManager parent) {
        return new DuelingQNetworkMLP(observations, actions, null, null, parent, DuelingType.MAX);
    }

    public static DuelingQNetworkMLP withType(int observations,
                                              int actions,
                                              NDManager parent,
                                              DuelingType duelingType) {
        return new DuelingQNetworkMLP(observations, actions, null, null, parent, duelingType);
    }

    public DuelingQNetworkMLP(int observations,
                              int actions,
                              NDManager parent,
                              DuelingType duelingType) {
        this(observations, actions, null, null, parent, duelingType);
    }

    @SneakyThrows
    public DuelingQNetworkMLP(int observations,
                              int actions,
                              Path modelPath,
                              String prefix,
                              NDManager parent,
                              @NonNull DuelingType duelingType) {
        this.observations = observations;
        this.actions = actions;
        this.subManager = subMgr(parent, getClass());
        this.model = newModel(getClass(), subManager.getDevice());
        this.net = new DuelingLayer(actions, duelingType);
        model.setBlock(net);

        this.parameterStore = new ParameterStore(subManager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(subManager, DataType.FLOAT32, new Shape(1, observations));
            DJLUtils.setGradients(model.getBlock());
            this.training = true;
        }
    }

    public DuelingType duelingType() {
        return net.duelingType();
    }

    @Override
    public NDList forward(NDList input) {
        return safeForwardSingle(subManager, net, parameterStore, input, training);
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
        var cloned = new DuelingQNetworkMLP(observations, actions, subManager, duelingType());
        setName(cloned.subManager, "clone");
        DJLUtils.copy(model.getBlock(), cloned.model.getBlock());
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
