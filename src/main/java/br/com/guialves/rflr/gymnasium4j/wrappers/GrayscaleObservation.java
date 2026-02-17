package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.GrayscaleObservation">...</a>
 *
 * @param keepDim
 */
public record GrayscaleObservation(boolean keepDim) implements IWrapper {

    public GrayscaleObservation() {
        this(false);
    }

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, keep_dim=%s)".formatted(this.getClass().getSimpleName(), varName, keepDim ? "True" : "False");
    }
}
