package br.com.guialves.rflr.dqn;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Slf4j
public class DeepQNetworkCNN implements AutoCloseable {

    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;
    private final boolean training;

    public DeepQNetworkCNN(int channels,
                           int size,
                           int actions,
                           NDManager manager) {
        this(channels, size, actions, null, null, manager);
    }

    @SneakyThrows
    public DeepQNetworkCNN(int channels,
                           int size,
                           int actions,
                           Path modelPath,
                           String prefix,
                           NDManager manager) {

        this.model = Model.newInstance("dqn_cnn");
        this.net = new SequentialBlock();

        net.add(Conv2d.builder()
                        .setFilters(32)
                        .setKernelShape(new Shape(8, 8))
                        .optStride(new Shape(4, 4))
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(4, 4))
                        .optStride(new Shape(2, 2))
                        .build())
                .add(Activation::relu)
                .add(Conv2d.builder()
                        .setFilters(64)
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .build())
                .add(Activation::relu)
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(512).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(actions).build());

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
                    new Shape(1, channels, size, size));
            this.training = true;
        }
    }

    public NDList forward(NDList input) {
        return net.forward(parameterStore, input, training);
    }

    public NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    public void save(Path path, String prefix) throws IOException {
        model.save(path, prefix);
    }

    @Override
    public void close() {
        model.close();
    }
}
