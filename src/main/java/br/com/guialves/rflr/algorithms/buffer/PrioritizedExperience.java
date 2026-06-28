package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;

public record PrioritizedExperience(NDArray state,
                                    ActionResult action,
                                    double reward,
                                    NDArray nextState,
                                    boolean done,
                                    float priority) implements IExperience {

}
