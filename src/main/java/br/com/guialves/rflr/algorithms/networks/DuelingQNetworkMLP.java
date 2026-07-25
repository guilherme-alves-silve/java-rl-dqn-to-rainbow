package br.com.guialves.rflr.algorithms.networks;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import br.com.guialves.rflr.djlutils.DJLUtils;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;
import java.util.function.BinaryOperator;

import static br.com.guialves.rflr.djlutils.DJLLayers.linear;

/**
 * Architecture based on the link below:
 * <a href="https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html">...</a>
 * For coding reference:
 * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-djl.html">...</a>
 * <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp-djl.html">...</a>
 */
@Slf4j
public class DuelingQNetworkMLP implements IDeepQNetwork {

    private boolean training;
    private final NDManager manager;
    private final int observations;
    private final int actions;
    private final Model model;
    private final DuelingNetworkBlock net;
    private final ParameterStore parameterStore;

    public enum DuelingType {
        MEAN, MAX
    }

    public static DuelingQNetworkMLP withMeanCalc(int observations,
                                                  int actions,
                                                  NDManager manager) {
        return new DuelingQNetworkMLP(observations, actions, null, null, manager, DuelingType.MEAN);
    }

    public static DuelingQNetworkMLP withMaxCalc(int observations,
                                                 int actions,
                                                 NDManager manager) {
        return new DuelingQNetworkMLP(observations, actions, null, null, manager, DuelingType.MAX);
    }

    public DuelingQNetworkMLP(int observations,
                              int actions,
                              NDManager manager,
                              DuelingType duelingType) {
        this(observations, actions, null, null, manager, duelingType);
    }

    @SneakyThrows
    public DuelingQNetworkMLP(int observations,
                              int actions,
                              Path modelPath,
                              String prefix,
                              NDManager manager,
                              DuelingType duelingType) {
        this.observations = observations;
        this.actions = actions;
        this.manager = manager;
        this.model = Model.newInstance("dueling_mlp", manager.getDevice());
        this.net = new DuelingNetworkBlock(actions, duelingType);
        model.setBlock(net);

        this.parameterStore = new ParameterStore(manager, false);
        if (modelPath != null) {
            log.info("Loading model: {}, {}", modelPath, prefix);
            model.load(modelPath, prefix);
            this.training = false;
        } else {
            net.initialize(manager, DataType.FLOAT32, new Shape(1, observations));
            DJLUtils.setGradients(model.getBlock());
            this.training = true;
        }
    }

    public DuelingType duelingType() {
        return net.duelingType;
    }

    @Override
    public NDList forward(NDList input) {
        return net.forward(parameterStore, input, training);
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
        var cloned = new DuelingQNetworkMLP(observations, actions, manager, device);
        DJLUtils.copy(model.getBlock(), cloned.model.getBlock());
        return cloned;
    }

    @Override
    public NDManager manager() {
        return this.manager;
    }

    @Override
    public void close() {
        model.close();
    }

    private static class DuelingNetworkBlock extends AbstractBlock {

        private final int actions;
        private final DuelingType duelingType;
        private final BinaryOperator<NDArray> qValueCalc;
        private final SequentialBlock featureBackbone;
        private final SequentialBlock valueHead;
        private final SequentialBlock advantageHead;

        /**
         * Q-Value calculator for Dueling DQN using the mean operation:
         *  Q(s, a) = V(s) + (A(s, a) - mean A(s, a'))
         */
        static final BinaryOperator<NDArray> Q_VALUE_MEAN = (value, advantage) -> {
            var subMean = advantage.subi(advantage.mean());
            return value.add(subMean);
        };

        /**
         * Q-Value calculator for Dueling DQN using the max operation:
         *  Q(s, a) = V(s) + (A(s, a) - max A(s, a'))
         */
        static final BinaryOperator<NDArray> Q_VALUE_MAX = (value, advantage) -> {
            var subMean = advantage.subi(advantage.max());
            return value.add(subMean);
        };

        public DuelingNetworkBlock(int actions, DuelingType duelingType) {

            this.actions = actions;
            this.duelingType = duelingType;
            this.qValueCalc = switch (duelingType) {
                case MEAN -> Q_VALUE_MEAN;
                case MAX -> Q_VALUE_MAX;
                default -> throw new UnsupportedOperationException("Type " + duelingType + " is not supported!");
            };
            this.featureBackbone = new SequentialBlock();
            featureBackbone.add(linear(128))
                    .add(Activation::relu)
                    .add(linear(128))
                    .add(Activation::relu);

            this.valueHead = new SequentialBlock();
            valueHead.add(linear(1));

            this.advantageHead = new SequentialBlock();
            advantageHead.add(linear(actions));

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
}
