package br.com.guialves.rflr.algorithms.networks.layers;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import lombok.Getter;
import lombok.experimental.Accessors;

import java.util.function.BinaryOperator;
import java.util.function.Consumer;

import static br.com.guialves.rflr.djlutils.DJLLayers.linear;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1_ARR;
import static br.com.guialves.rflr.djlutils.DJLUtils.KEEP_DIMS;

@Accessors(fluent = true)
public class DuelingLayer extends AbstractBlock {

    private static final byte VERSION = 1;

    protected final int actions;
    protected final BinaryOperator<NDArray> qValueCalc;
    protected final SequentialBlock featureBackbone;
    protected final SequentialBlock valueHead;
    protected final SequentialBlock advantageHead;

    /**
     * Q-Value calculator for Dueling DQN using the mean operation:
     *  Q(s, a) = V(s) + (A(s, a) - mean A(s, a'))
     */
    private static final BinaryOperator<NDArray> Q_VALUE_MEAN = (value, advantage) -> {
        var subMean = advantage.sub(advantage.mean(AXIS_1_ARR, KEEP_DIMS));
        return value.add(subMean);
    };

    /**
     * Q-Value calculator for Dueling DQN using the max operation:
     *  Q(s, a) = V(s) + (A(s, a) - max A(s, a'))
     */
    private static final BinaryOperator<NDArray> Q_VALUE_MAX = (value, advantage) -> {
        var subMean = advantage.sub(advantage.max(AXIS_1_ARR, KEEP_DIMS));
        return value.add(subMean);
    };

    @Getter
    private final DuelingType duelingType;

    public DuelingLayer(int actions,
                        DuelingType duelingType) {
        Consumer<SequentialBlock> featureBackboneModifier = featureBackbone -> featureBackbone.add(linear(128))
                .add(Activation::relu)
                .add(linear(128))
                .add(Activation::relu);
        Consumer<SequentialBlock> valueModifier = valueHead -> valueHead.add(linear(1));
        Consumer<SequentialBlock> advantageModifier = advantageHead -> advantageHead.add(linear(actions));
        this(actions, duelingType, featureBackboneModifier, valueModifier, advantageModifier);
    }

    public DuelingLayer(int actions,
                        DuelingType duelingType,
                        Consumer<SequentialBlock> featureBackboneModifier,
                        Consumer<SequentialBlock> valueHeadModifier,
                        Consumer<SequentialBlock> advantageHeadModifier) {
        super(VERSION);
        this.actions = actions;
        this.duelingType = duelingType;
        this.qValueCalc = switch (duelingType) {
            case MEAN -> Q_VALUE_MEAN;
            case MAX -> Q_VALUE_MAX;
        };
        this.featureBackbone = new SequentialBlock();
        featureBackboneModifier.accept(featureBackbone);

        this.valueHead = new SequentialBlock();
        valueHeadModifier.accept(valueHead);

        this.advantageHead = new SequentialBlock();
        advantageHeadModifier.accept(advantageHead);

        addChildBlock("featureBackbone", featureBackbone);
        addChildBlock("valueHead", valueHead);
        addChildBlock("advantageHead", advantageHead);
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        featureBackbone.initialize(manager, dataType, inputShapes);
        var featureOutShapes = featureBackbone.getOutputShapes(inputShapes);
        valueHead.initialize(manager, dataType, featureOutShapes);
        advantageHead.initialize(manager, dataType, featureOutShapes);
    }

    @Override
    public NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        if (inputs.size() != 1) throw new IllegalArgumentException("The dueling dqn just accepts one input!");
        var feature = featureBackbone.forward(parameterStore, inputs, training);
        var value = valueHead.forward(parameterStore, feature, training).singletonOrThrow();
        var advantage = advantageHead.forward(parameterStore, feature, training).singletonOrThrow();
        return new NDList(qValueCalc.apply(value, advantage));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {new Shape(inputShapes[0].get(0), actions)};
    }
}
