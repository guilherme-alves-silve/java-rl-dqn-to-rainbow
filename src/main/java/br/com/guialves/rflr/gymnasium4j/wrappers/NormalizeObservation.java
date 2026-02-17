package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.NormalizeObservation">...</a>
 *
 * @param epsilon
 */
public record NormalizeObservation(float epsilon) implements IWrapper {

    public NormalizeObservation() {
        this(1e-8f);
    }

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, epsilon=%s)".formatted(this.getClass().getSimpleName(), varName, epsilon);
    }
}
