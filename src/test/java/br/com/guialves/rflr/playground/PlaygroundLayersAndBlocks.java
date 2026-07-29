package br.com.guialves.rflr.playground;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import lombok.Cleanup;
import lombok.NonNull;

/**
 * This part is to test and learn more about the custom block in DJL, used
 * as base for the Noisy Networks
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/index.html">Deep Learning Computation</a>
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/model-construction.html">Layers and Blocks</a>
 */
public class PlaygroundLayersAndBlocks {

    static void main() throws TranslateException {
        @Cleanup var manager = NDManager.newBaseManager();
        int inputSize = 20;
        var x = manager.randomUniform(0f, 1f, new Shape(2, inputSize));
        @Cleanup var model = Model.newInstance("lin-reg");

        var net = new SequentialBlock();
        net.add(Linear.builder().setUnits(256).build())
           .add(Activation.reluBlock())
           .add(Linear.builder().setUnits(10).build())
           .setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, x.getShape());
        model.setBlock(net);

        var translator = new NoopTranslator();
        var xList = new NDList(x);
        @Cleanup Predictor<NDList, NDList> predictor = model.newPredictor(translator);
        var out = predictor.predict(xList).singletonOrThrow();
        IO.println("out: " + out);

        var mlp = new MLP(inputSize);
        mlp.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        mlp.initialize(manager, DataType.FLOAT32, x.getShape());
        model.setBlock(mlp);

        @Cleanup Predictor<NDList, NDList> mlpPredictor = model.newPredictor(translator);
        IO.println("mlp-out: " + mlpPredictor.predict(xList).singletonOrThrow());

        var seqNet = new MySequential()
                .add(Linear.builder().setUnits(256).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(10).build());

        seqNet.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        seqNet.initialize(manager, DataType.FLOAT32, x.getShape());

        @Cleanup var seqModel = Model.newInstance("my-sequential");
        seqModel.setBlock(seqNet);

        @Cleanup Predictor<NDList, NDList> seqPredictor = seqModel.newPredictor(translator);
        IO.println("seq out: " + seqPredictor.predict(xList).singletonOrThrow());

        var netFixed = new FixedHiddenMLP();

        netFixed.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        netFixed.initialize(manager, DataType.FLOAT32, x.getShape());

        @Cleanup var modelFixed = Model.newInstance("fixed-mlp");
        modelFixed.setBlock(netFixed);

        @Cleanup Predictor<NDList, NDList> fixedPredictor = modelFixed.newPredictor(translator);
        IO.println("fixed out: " + fixedPredictor.predict(xList).singletonOrThrow());

        var chimera = new SequentialBlock();
        chimera.add(new NestMLP())
               .add(Linear.builder().setUnits(20).build())
               .add(new FixedHiddenMLP())
               .setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        chimera.initialize(manager, DataType.FLOAT32, x.getShape());
        @Cleanup var modelChimera = Model.newInstance("chimera");
        modelChimera.setBlock(chimera);

        @Cleanup Predictor<NDList, NDList> chimeraPredictor = modelChimera.newPredictor(translator);
        IO.println("chimera out: " + chimeraPredictor.predict(xList).singletonOrThrow());
    }

    private static class MLP extends AbstractBlock {

        private static final byte VERSION = 1;

        private final int inputSize;
        private final Block flattenInput;
        private final Block hidden256;
        private final Block output10;

        public MLP(int inputSize) {
            super(VERSION);
            this.inputSize = inputSize;
            this.flattenInput = addChildBlock("flattenInput", Blocks.batchFlattenBlock(inputSize));
            this.hidden256 = addChildBlock("hidden256", Linear.builder().setUnits(256).build());
            this.output10 = addChildBlock("output10", Linear.builder().setUnits(10).build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore,
                                         NDList inputs,
                                         boolean training,
                                         PairList<String, Object> params) {
            NDList current = inputs;
            current = flattenInput.forward(parameterStore, current, training);
            current = hidden256.forward(parameterStore, current, training);
            current = Activation.relu(current);
            current = output10.forward(parameterStore, current, training);
            return current;
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputs) {
            var current = inputs;
            for (var block : children.values()) {
                current = block.getOutputShapes(current);
            }
            return current;
        }

        @Override
        public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            hidden256.initialize(manager, dataType, new Shape(1, inputSize));
            output10.initialize(manager, dataType, new Shape(1, 256));
        }
    }

    private static class MySequential extends AbstractBlock {

        private static final byte VERSION = 2;

        public MySequential() {
            super(VERSION);
        }

        public MySequential add(@NonNull Block block) {
            addChildBlock(block.getClass().getSimpleName(), block);
            return this;
        }


        @Override
        protected NDList forwardInternal(ParameterStore parameterStore,
                                         NDList inputs,
                                         boolean training,
                                         PairList<String, Object> params) {
            var current = inputs;
            for (var block : children.values()) {
                current = block.forward(parameterStore, current, training, params);
            }
            return current;
        }

        @Override
        protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            var shapes = inputShapes;
            for (var child : children.values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            var current = inputShapes;
            for (var block : getChildren().values()) {
                current = block.getOutputShapes(current);
            }
            return current;
        }
    }

    /**
     * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/model-construction.html#executing-code-in-the-forward-method">Reference</a>
     */
    private static class FixedHiddenMLP extends AbstractBlock {

        private static final byte VERSION = 1;

        private final Block hidden20;
        private NDArray constantParamWeight;
        private NDArray constantParamBias;

        public FixedHiddenMLP() {
            super(VERSION);
            this.hidden20 = addChildBlock("denseLayer", Linear.builder().setUnits(20).build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
            var current = inputs;
            current = hidden20.forward(parameterStore, current, training, params);
            // e.g.: xW.T + b
            current = Linear.linear(current.singletonOrThrow(), constantParamWeight, constantParamBias);
            current = new NDList(Activation.relu(current.singletonOrThrow()));
            current = hidden20.forward(parameterStore, current, training);
            // not useful (probably), used just to show that you can do many things
            while (current.head().abs().sum().getFloat() > 1) {
                current.head().divi(2);
            }

            return new NDList(current.head().abs().sum());
        }

        @Override
        protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            var shapes = inputShapes;
            for (var child : children.values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }

            constantParamWeight = manager.randomUniform(-0.07f, 0.07f, new Shape(20, 20));
            constantParamBias = manager.zeros(new Shape(20));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[] {new Shape(1)};
        }
    }

    private static class NestMLP extends AbstractBlock {

        private static final byte VERSION = 1;

        private final SequentialBlock net;
        private final Block dense;

        public NestMLP() {
            super(VERSION);
            this.net = new SequentialBlock();
            net.add(Linear.builder().setUnits(64).build())
               .add(Activation.reluBlock())
               .add(Linear.builder().setUnits(32).build())
               .add(Activation.reluBlock());
            addChildBlock("net", net);
            this.dense = addChildBlock("dense", Linear.builder().setUnits(64).build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore,
                                         NDList inputs,
                                         boolean training,
                                         PairList<String, Object> params) {
            var current = inputs;
            current = net.forward(parameterStore, current, training);
            current = dense.forward(parameterStore, current, training);
            current = new NDList(Activation.relu(current.singletonOrThrow()));
            return current;
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputs) {
            var current = inputs;
            for (var block : children.values()) {
                current = block.getOutputShapes(current);
            }
            return current;
        }

        @Override
        protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            var shapes = inputShapes;
            for (var child : children.values()) {
                child.initialize(manager, dataType, shapes);
                shapes = child.getOutputShapes(shapes);
            }
        }
    }
}
