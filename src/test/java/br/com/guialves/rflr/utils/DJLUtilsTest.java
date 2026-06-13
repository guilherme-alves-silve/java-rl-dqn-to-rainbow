package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DJLUtilsTest {

    @Test
    void testCopy() {
        try (var manager = NDManager.newBaseManager()) {
            var net1 = new SequentialBlock()
                    .add(Linear.builder()
                            .setUnits(128)
                            .optBias(true)
                            .build())
                    .add(Activation::relu);
            net1.initialize(manager, DataType.FLOAT32, new Shape(2, 2));

            var net2 = new SequentialBlock()
                    .add(Linear.builder()
                            .setUnits(128)
                            .optBias(true)
                            .build())
                    .add(Activation::relu);
            net2.initialize(manager, DataType.FLOAT32, new Shape(2, 2));

            assertTrue(DJLUtils.diff(net1, net2));
            DJLUtils.copy(net1, net2);
            assertFalse(DJLUtils.diff(net1, net2));
        }
    }
}