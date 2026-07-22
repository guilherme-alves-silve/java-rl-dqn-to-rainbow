package br.com.guialves.rflr.algorithms.buffer;

import ai.djl.ndarray.NDArray;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType.ActionResult;
import lombok.With;

/**
 * Record used in the PER algorithm. The object is immutable, you can create a new object but
 * with a different priority using @With.
 * Reference:
 *  <a href="https://medium.com/@jmgoyesc/enhancing-java-records-with-lombok-combining-simplicity-and-flexibility-for-immutability-982db40daa95">Java records</a>
 */
public record PrioritizedExperience(NDArray state,
                                    ActionResult action,
                                    double reward,
                                    NDArray nextState,
                                    boolean done,
                                    @With float priority) implements IExperience {

}
