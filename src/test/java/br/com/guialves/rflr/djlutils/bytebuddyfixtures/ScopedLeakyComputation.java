package br.com.guialves.rflr.djlutils.bytebuddyfixtures;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.djlutils.DJLScoped;

public class ScopedLeakyComputation {

    @DJLScoped
    public float calculate(NDManager manager) {
        var input = manager.ones(new Shape(2, 2));
        var shifted = input.add(1);
        var scaled = shifted.mul(2);
        var mean = scaled.mean();
        return mean.getFloat();
    }
}
