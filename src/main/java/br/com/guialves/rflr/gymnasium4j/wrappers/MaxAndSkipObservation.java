package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.MaxAndSkipObservation">...</a>
 *
 * @param skip
 */
public record MaxAndSkipObservation(int skip) implements IWrapper {

    public MaxAndSkipObservation() {
        this(4);
    }

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, skip=%s)".formatted(this.getClass().getSimpleName(), varName, skip);
    }
}
