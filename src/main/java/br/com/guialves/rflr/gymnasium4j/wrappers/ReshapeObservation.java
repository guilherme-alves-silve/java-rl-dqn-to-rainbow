package br.com.guialves.rflr.gymnasium4j.wrappers;

import java.util.Arrays;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.ReshapeObservation">...</a>
 *
 * @param shape
 */
public record ReshapeObservation(int[] shape) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, shape=%s)".formatted(this.getClass().getSimpleName(),
                varName, Arrays.toString(shape));
    }
}
