package br.com.guialves.rflr.gymnasium4j.wrappers;

import java.util.Arrays;

public record ReshapeObservation(int[] shape) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, shape=%s)".formatted(this.getClass().getSimpleName(),
                varName, Arrays.toString(shape));
    }
}
