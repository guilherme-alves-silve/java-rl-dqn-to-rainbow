package br.com.guialves.rflr.gymnasium4j.dto;

import java.util.Arrays;

public record EnvRenderMetadata(int[] shape, String dtype) {

    public int height() {
        return shape[0];
    }

    public int width() {
        return shape[1];
    }

    public int channels() {
        return shape[2];
    }

    public int size() {
        return Arrays.stream(shape).reduce(1, ((left, right) -> left * right));
    }
}
