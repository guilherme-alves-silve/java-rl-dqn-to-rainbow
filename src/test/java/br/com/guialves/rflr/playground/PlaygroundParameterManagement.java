package br.com.guialves.rflr.playground;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;
import lombok.Cleanup;

import java.util.stream.IntStream;

/**
 * This part is to test and learn more about the parameters in DJL, used
 * as base for the Noisy Networks
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/index.html">Deep Learning Computation</a>
 * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/parameters.html">Parameter Management</a>
 */
public class PlaygroundParameterManagement {

    static void main() {
        @Cleanup var manager = NDManager.newBaseManager();
        var x = manager.randomUniform(0, 1, new Shape(2, 4));
        var net = new SequentialBlock()
                .add(Linear.builder().setUnits(8).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(1).build());
        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, x.getShape());

        var ps = new ParameterStore(manager, false);
        var out = net.forward(ps, new NDList(x), false).head();
        IO.println("out: " + out);

        var params = net.getParameters();
        for (var pair : params) {
            IO.println(pair.getKey());
        }

        // by name
        var dense0Weight = params.get("01Linear_weight").getArray();
        var dense0Bias = params.get("01Linear_bias").getArray();
        // by index
        var dense1Weight = params.valueAt(2).getArray();
        var dense1Bias = params.valueAt(3).getArray();

        IO.println(dense0Weight);
        IO.println(dense0Bias);
        IO.println(dense1Weight);
        IO.println(dense1Bias);

        IO.println("dense0Weight.gradient: " + dense0Weight.getGradient());

        // not good practice, but I'm just testing
        class NetBuilder {
            SequentialBlock block1() {
                return new SequentialBlock()
                    .add(Linear.builder().setUnits(32).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(16).build())
                    .add(Activation.reluBlock());
            }

            SequentialBlock block2() {
                var net = new SequentialBlock();
                IntStream.range(0, 4).forEach(_ -> net.add(block1()));
                return net;
            }
        }

        var rgnet = new SequentialBlock()
                .add(new NetBuilder().block2())
                .add(Linear.builder().setUnits(10).build());
        rgnet.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        rgnet.initialize(manager, DataType.FLOAT32, x.getShape());
        var out2 = rgnet.forward(ps, new NDList(x), false).singletonOrThrow();
        IO.println("out2: " + out2);
        IO.println("rgnet: " + rgnet);

        for (var param : rgnet.getParameters()) {
            IO.println(param.getValue().getArray());
        }

        var majorBlock = rgnet.getChildren().get(0).getValue();
        var subBlock2 = majorBlock.getChildren().valueAt(1);
        var linearLayer1 = subBlock2.getChildren().valueAt(0);
        NDArray bias = linearLayer1.getParameters().valueAt(1).getArray();
        IO.println(bias);

        // another initializer won't work when we already set one
        net.setInitializer(new ConstantInitializer(1), Parameter.Type.WEIGHT);
        net.initialize(manager, DataType.FLOAT32, x.getShape());

        var linearLayer = net.getChildren().get(0).getValue();
        var weight = linearLayer.getParameters().get(0).getValue().getArray();
        IO.println("weight: " + weight);

        class NetBuilder2 {
            SequentialBlock getNet() {
                return new SequentialBlock()
                        .add(Linear.builder().setUnits(8).build())
                        .add(Activation.reluBlock())
                        .add(Linear.builder().setUnits(1).build());
            }
        }

        // now all initialized with luck 777
        var net2 = new NetBuilder2().getNet();
        net2.setInitializer(new ConstantInitializer(777), Parameter.Type.WEIGHT);
        net2.initialize(manager, DataType.FLOAT32, x.getShape());

        var linearLayer2 = net2.getChildren().valueAt(0);
        var weight2 = linearLayer2.getParameters().get(0).getValue().getArray();
        IO.println("weight2: " + weight2);

        // gaussian std dev 0.01
        var net3 = new NetBuilder2().getNet();
        net3.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        net3.initialize(manager, DataType.FLOAT32, x.getShape());
        var linearLayer3 = net3.getChildren().valueAt(0);
        var weight3 = linearLayer3.getParameters().get(0).getValue().getArray();
        IO.println("weight3: " + weight3);

        var net4 = new SequentialBlock();
        var linear1 = Linear.builder().setUnits(8).build();
        net4.add(linear1)
            .add(Activation.reluBlock());
        var linear2 = Linear.builder().setUnits(1).build();
        net4.add(linear2);

        linear1.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        linear1.initialize(manager, DataType.FLOAT32, x.getShape());

        linear2.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
        linear2.initialize(manager, DataType.FLOAT32, x.getShape());

        IO.println("init with xavier: " + linear1.getParameters().valueAt(0).getArray());
        IO.println("init with zeros: " + linear2.getParameters().valueAt(0).getArray());

        var net5 = new NetBuilder2().getNet();
        var params2 = net5.getParameters();
        params2.get("01Linear_weight").setInitializer(new NormalInitializer());
        params2.get("03Linear_weight").setInitializer(Initializer.ONES);
        net5.initialize(manager, DataType.FLOAT32, new Shape(2, 4));
        IO.println("with normal init - " + params2.valueAt(0).getArray());
        IO.println("with one init - " + params2.valueAt(2).getArray());

        var net6 = new NetBuilder2().getNet();
        net6.setInitializer(new MyInit(), Parameter.Type.WEIGHT);
        net6.initialize(manager, DataType.FLOAT32, x.getShape());
        var linearLayer6 = net6.getChildren().valueAt(0);
        var weight6 = linearLayer6.getParameters().valueAt(0).getArray();
        IO.println("with my init - " + weight6);

        // '__'i() inplace operator modify original array
        var weightLayer = net.getChildren().valueAt(0)
                .getParameters().valueAt(0).getArray();
        weightLayer.addi(7);
        weightLayer.divi(9);
        weightLayer.set(new NDIndex(0, 0), 2020);
        IO.println("modifiy inplace with 'i' - " + weightLayer);

        // Tied parameters, shared layers, (e.g. used when sharing encoding and decoding words)
        var net7 = new SequentialBlock();
        var shared = Linear.builder().setUnits(8).build();
        var sharedRelu = new SequentialBlock()
                .add(shared)
                .add(Activation.reluBlock());
        net7.add(Linear.builder().setUnits(8).build())
                .add(Activation.reluBlock())
                .add(sharedRelu)
                .add(sharedRelu)
                .add(Linear.builder().setUnits(10).build())
                .setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
        var newX = manager.randomUniform(-10f, 10f, new Shape(2, 20), DataType.FLOAT32);
        net7.initialize(manager, DataType.FLOAT32, newX.getShape());
        var outNewX = net7.forward(ps, new NDList(newX), false).singletonOrThrow();
        IO.println("outNewX: " + outNewX);

        // Check that the parameters are the same
        var shared1 = net7.getChildren().valueAt(2)
                .getParameters().valueAt(0).getArray();
        var shared2 = net7.getChildren().valueAt(3)
                .getParameters().valueAt(0).getArray();
        IO.println("Same? " + shared1.eq(shared2));
    }

    /**
     * <a href="https://d2l.djl.ai/chapter_deep-learning-computation/parameters.html#custom-initialization">Custom Init</a>
     */
    private static class MyInit implements Initializer {

        @Override
        public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
            System.out.printf("Init %s\n", shape.toString());
            var data = manager.randomUniform(-10, 10, shape, dataType);
            var absGte5 = data.abs().gte(5);
            return data.mul(absGte5);
        }
    }
}
