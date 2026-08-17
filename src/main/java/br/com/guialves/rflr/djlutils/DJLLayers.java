package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;

public class DJLLayers {

    private DJLLayers() {
        throw new IllegalStateException("No DJLLayers!");
    }

    public static Linear linear(long units) {
        return linear(units, true);
    }

    public static Linear linear(long units, boolean bias) {
        return Linear.builder()
                .setUnits(units)
                .optBias(bias)
                .build();
    }

    public static Conv2d conv2d(int filters, int shape, int stride) {
        return Conv2d.builder()
                .setFilters(filters)
                .setKernelShape(new Shape(shape, shape))
                .optStride(new Shape(stride, stride))
                .build();
    }

    public static Dropout dropout(float rate) {
        return Dropout.builder()
                .optRate(rate)
                .build();
    }
}
