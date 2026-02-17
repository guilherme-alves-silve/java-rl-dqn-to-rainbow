package br.com.guialves.rflr.gymnasium4j.wrappers;

public record DelayObservation(int delay) implements IWrapper {

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, delay=%d)".formatted(this.getClass().getSimpleName(), varName, delay);
    }
}
