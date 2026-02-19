package br.com.guialves.rflr.dqn;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;

/**
 * Architecture based on the link below:
 * <a href="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">...</a>
 * For coding reference:
 * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-djl.html">...</a>
 * <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp-djl.html">...</a>
 */
@Slf4j
public class DeepQNetworkMLP implements AutoCloseable {

    private final Model model;
    private final SequentialBlock net;
    private final ParameterStore parameterStore;
    private final boolean training;

    public DeepQNetworkMLP(int observations,
                           int actions,
                           NDManager manager) {
        this(observations, actions, null, null, manager);
    }

    @SneakyThrows
    public DeepQNetworkMLP(int observations,
                           int actions,
                           Path modelPath,
                           String prefix,
                           NDManager manager) {
        this.model = Model.newInstance("dqn_mlp");
        this.net = new SequentialBlock();
        net.add(Linear.builder().setUnits(128).optBias(true).build())
            .add(Activation::relu)
            .add(Linear.builder().setUnits(128).optBias(true).build())
            .add(Activation::relu)
            .add(Linear.builder().setUnits(actions).optBias(true).build());
        model.setBlock(net);

        this.parameterStore = new ParameterStore(manager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(manager, DataType.FLOAT32, new Shape(1, observations));
            this.training = true;
        }
    }

    public NDList forward(NDList input) {
        return net.forward(parameterStore, input, training);
    }

    public NDArray forward(NDArray input) {
        return forward(new NDList(input)).singletonOrThrow();
    }

    @SneakyThrows
    public void save(Path modelPath, String newModelName) {
        this.model.save(modelPath, newModelName);
    }

    @Override
    public void close() {
        model.close();
    }
}
