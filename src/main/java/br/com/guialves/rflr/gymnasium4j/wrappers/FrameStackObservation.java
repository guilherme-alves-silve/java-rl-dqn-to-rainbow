package br.com.guialves.rflr.gymnasium4j.wrappers;

/**
 * Equivalent to the Gymnasium (from Python):
 *  <a href="https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStackObservation">...</a>
 *
 * @param stackSize
 */
public record FrameStackObservation(int stackSize) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, stack_size=%s)".formatted(this.getClass().getSimpleName(), varName, stackSize);
    }
}
