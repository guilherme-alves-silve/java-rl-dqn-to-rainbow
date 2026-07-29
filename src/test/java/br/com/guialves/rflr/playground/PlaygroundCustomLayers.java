package br.com.guialves.rflr.playground;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import lombok.Cleanup;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.newModel;

/**
 * This part is to test and learn more about the parameters in DJL, used
 * as base for the Noisy Networks
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/index.html">Deep Learning Computation</a>
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/custom-layer.html">Custom Layer</a>
 */
public class PlaygroundCustomLayers {

    static void main() throws TranslateException {
        @Cleanup var manager = NDManager.newBaseManager();

        var linear = new MyLinear(3, 5);
        var params = linear.getParameters();
        for (var param : params) {
            System.out.println(param.getKey());
        }

        var input = manager.randomUniform(0, 1, new Shape(2, 5));
        linear.initialize(manager, DataType.FLOAT32, input.getShape());

        @Cleanup var model = newModel("my-linear");
        model.setBlock(linear);

        Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
        var out = predictor.predict(new NDList(input)).singletonOrThrow();
        IO.println("out: " + out);
    }

    private static class MyLinear extends AbstractBlock {

        private final int inUnits;
        private final int outUnits;
        private final Parameter weight;
        private final Parameter bias;

        // outUnits: the number of outputs in this layer
        // inUnits: the number of inputs in this layer
        public MyLinear(int outUnits, int inUnits) {
            this.inUnits = inUnits;
            this.outUnits = outUnits;
            weight = addParameter(
                    Parameter.builder()
                            .setName("weight")
                            .setType(Parameter.Type.WEIGHT)
                            .optShape(new Shape(inUnits, outUnits))
                            .build());
            bias = addParameter(
                    Parameter.builder()
                            .setName("bias")
                            .setType(Parameter.Type.BIAS)
                            .optShape(new Shape(outUnits))
                            .build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore parameterStore,
                                         NDList inputs,
                                         boolean training,
                                         PairList<String, Object> params) {
            var input = inputs.singletonOrThrow();
            var device = input.getDevice();
            // Since we added the parameter, we can now access it from the parameter store
            var weightArr = parameterStore.getValue(weight, device, false);
            var biasArr = parameterStore.getValue(bias, device, false);
            return relu(linear(input, weightArr, biasArr));
        }

        public static NDArray linear(NDArray input, NDArray weight, NDArray bias) {
            return input.dot(weight).add(bias);
        }

        public static NDList relu(NDArray input) {
            return new NDList(Activation.relu(input));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputs) {
            return new Shape[]{new Shape(outUnits, inUnits)};
        }
    }
}
