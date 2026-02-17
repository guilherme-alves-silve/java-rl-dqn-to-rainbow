package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.DelayObservation">...</a>
 *
 * @param delay
 */
public record DelayObservation(int delay) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, delay=%d)".formatted(this.getClass().getSimpleName(), varName, delay);
    }
}
