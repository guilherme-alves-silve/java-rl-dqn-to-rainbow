package br.com.guialves.rflr.djlutils.bytebuddyfixtures;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.djlutils.DJLScoped;

public class ScopedReturningComputation {

    @DJLScoped
    public NDArray calculate(NDManager manager) {
        var input = manager.ones(new Shape(1));
        var shifted = input.add(1);
        return shifted.mul(2);
    }
}
