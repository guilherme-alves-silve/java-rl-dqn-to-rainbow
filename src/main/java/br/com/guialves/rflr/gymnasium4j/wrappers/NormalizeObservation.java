package br.com.guialves.rflr.gymnasium4j.wrappers;

public record NormalizeObservation(float epsilon) implements IWrapper {

    public NormalizeObservation() {
        this(1e-8f);
    }

    @Override
    public String pyToStr(String varName) {
        return "%s(%s, epsilon=%s)".formatted(this.getClass().getSimpleName(), varName, epsilon);
    }
}
