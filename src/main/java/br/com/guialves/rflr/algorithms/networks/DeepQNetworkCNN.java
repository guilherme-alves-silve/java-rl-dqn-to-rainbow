package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import br.com.guialves.rflr.djlutils.DJLUtils;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;

import static br.com.guialves.rflr.djlutils.DJLLayers.conv2d;
import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.safeForwardSingle;

@Slf4j
public class DeepQNetworkCNN implements IDeepQNetwork {

    private boolean training;
    private final NDManager manager;
    private final int channels;
    private final int observations;
    private final int actions;
    private final Path modelPath;
    private final String prefix;
    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;

    public DeepQNetworkCNN(int channels,
                           int observations,
                           int actions,
                           NDManager manager) {
        this(channels, observations, actions, null, null, manager);
    }

    @SneakyThrows
    public DeepQNetworkCNN(int channels,
                           int observations,
                           int actions,
                           Path modelPath,
                           String prefix,
                           NDManager manager) {
        this.channels = channels;
        this.observations = observations;
        this.actions = actions;
        this.modelPath = modelPath;
        this.prefix = prefix;
        this.manager = manager;
        this.model = Model.newInstance("dqn_cnn", manager.getDevice());
        this.net = new SequentialBlock();

        net.add(conv2d(32, 8, 4))
                .add(Activation::relu)
                .add(conv2d(64, 4, 2))
                .add(Activation::relu)
                .add(conv2d(64, 3, 1))
                .add(Activation::relu)
                .add(Blocks.batchFlattenBlock())
                .add(linear(512))
                .add(Activation::relu)
                .add(linear(actions));

        model.setBlock(net);

        this.parameterStore = new ParameterStore(manager, false);

        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            // Atari input: (batch, channels, height, width)
            net.initialize(manager,
                    DataType.FLOAT32,
                    new Shape(1, channels, observations, observations));
            DJLUtils.setGradients(model.getBlock());
            this.training = true;
        }
    }

    @Override
    public NDList forward(NDList input) {
        return safeForwardSingle(manager, net, parameterStore, input, training);
    }

    @Override
    public NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    @Override
    public NDManager manager() {
        return this.manager;
    }

    @Override
    @SneakyThrows
    public void save(Path path, String prefix) {
        model.save(path, prefix);
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
        var cloned = new DeepQNetworkCNN(channels, observations, actions,
                                         modelPath, prefix, manager);
        DJLUtils.copy(model.getBlock(), cloned.model.getBlock());
        return cloned;
    }

    @Override
    public void close() {
        model.close();
    }
}
