package br.com.guialves.rflr.gymnasium4j.wrappers;

public interface IWrapper {

    default String pyToStr(String varName) {
        return "%s(%s)".formatted(this.getClass().getSimpleName(), varName);
    }
}
