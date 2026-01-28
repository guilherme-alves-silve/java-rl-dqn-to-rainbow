package br.com.guialves.rflr.gymnasium4j.dto;

import ai.djl.ndarray.NDArray;

public record Experience(NDArray state,
                         NDArray action,
                         NDArray reward,
                         NDArray nextState,
                         NDArray done) {

}
