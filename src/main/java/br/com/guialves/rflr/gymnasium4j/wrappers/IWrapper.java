package br.com.guialves.rflr.gymnasium4j.wrappers;

import java.util.Arrays;

import static java.util.stream.Collectors.joining;

public interface IWrapper {

    default String pyToStr(String varName) {
        return "%s(%s)".formatted(this.getClass().getSimpleName(), varName);
    }

    String pyKwargs();

    default String toPyTupleStr(int[] array) {
        return Arrays.stream(array)
                .mapToObj(Integer::toString)
                .collect(joining(",", "(", ")"));
    }
}
