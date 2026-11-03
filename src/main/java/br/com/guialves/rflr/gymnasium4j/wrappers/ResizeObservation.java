package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.ResizeObservation">...</a>
 *
 * @param shape
 */
public record ResizeObservation(int[] shape) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, shape=%s)".formatted(this.getClass().getSimpleName(),
                varName, toPyTupleStr(shape));
    }

    @Override
    public String pyKwargs() {
        return "{'shape': %s}".formatted(toPyTupleStr(shape));
    }
}
