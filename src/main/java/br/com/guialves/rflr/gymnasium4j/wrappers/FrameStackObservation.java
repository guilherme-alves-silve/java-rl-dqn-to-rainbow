package br.com.guialves.rflr.gymnasium4j.wrappers;

public record FrameStackObservation(int stackSize) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, stack_size=%s)".formatted(this.getClass().getSimpleName(), varName, stackSize);
    }
}
