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
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;

import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.copy;
import static br.com.guialves.rflr.djlutils.DJLUtils.setGradients;

/**
 * Architecture based on the link below:
 * <a href="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">...</a>
 * For coding reference:
 * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-djl.html">...</a>
 * <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp-djl.html">...</a>
 */
@Slf4j
public class DeepQNetworkMLP implements IDeepQNetwork {

    private boolean training;
    private final NDManager subManager;
    private final int observations;
    private final int actions;
    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;

    public DeepQNetworkMLP(int observations,
                           int actions,
                           NDManager parent) {
        this(observations, actions, null, null, parent);
    }

    @SneakyThrows
    public DeepQNetworkMLP(int observations,
                           int actions,
                           Path modelPath,
                           String prefix,
                           NDManager parent) {
        this.observations = observations;
        this.actions = actions;
        this.subManager = subMgr(parent, getClass());
        this.model = newModel(getClass(), subManager.getDevice());
        this.net = new SequentialBlock();
        net.add(linear(128))
           .add(Activation::relu)
           .add(linear(128))
           .add(Activation::relu)
           .add(linear(actions));
        model.setBlock(net);

        this.parameterStore = new ParameterStore(subManager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(subManager, DataType.FLOAT32, new Shape(1, observations));
            setGradients(model.getBlock());
            this.training = true;
        }
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
        var cloned = new DeepQNetworkMLP(observations, actions, subManager);
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
