package br.com.guialves.rflr.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class DeepQNetwork extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int numActions;
    private final int inputChannels;

    private Block conv1, conv2, conv3;
    private Block fc1, fc2;

    public DeepQNetwork(int inputChannels, int numActions) {
        super(VERSION);
        this.inputChannels = inputChannels;
        this.numActions = numActions;

        conv1 = addChildBlock("conv1",
                Conv2d.builder()
                        .setKernelShape(new Shape(8, 8))
                        .setFilters(32)
                        .optStride(new Shape(4, 4))
                        .build());

        conv2 = addChildBlock("conv2",
                Conv2d.builder()
                        .setKernelShape(new Shape(4, 4))
                        .setFilters(64)
                        .optStride(new Shape(2, 2))
                        .build());

        conv3 = addChildBlock("conv3",
                Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .setFilters(64)
                        .optStride(new Shape(1, 1))
                        .build());

        fc1 = addChildBlock("fc1", Linear.builder().setUnits(512).build());
        fc2 = addChildBlock("fc2", Linear.builder().setUnits(numActions).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore paramStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();

        x = conv1.forward(paramStore, new NDList(x), training, params).singletonOrThrow();
        x = Activation.relu(x);

        x = conv2.forward(paramStore, new NDList(x), training, params).singletonOrThrow();
        x = Activation.relu(x);

        x = conv3.forward(paramStore, new NDList(x), training, params).singletonOrThrow();
        x = Activation.relu(x);

        var shape = x.getShape();
        x = x.reshape(shape.get(0), -1);

        x = fc1.forward(paramStore, new NDList(x), training, params).singletonOrThrow();
        x = Activation.relu(x);

        x = fc2.forward(paramStore, new NDList(x), training, params).singletonOrThrow();

        return new NDList(x);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        long batchSize = inputShapes[0].get(0);
        return new Shape[]{new Shape(batchSize, numActions)};
    }
}
