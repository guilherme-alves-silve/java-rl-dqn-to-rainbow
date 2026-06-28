package br.com.guialves.rflr.algorithms;

import ai.djl.ndarray.NDArray;
import ai.djl.training.loss.Loss;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.algorithms.buffer.ExperienceReplayBuffer;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public interface IAgent {

    static final int DEFAULT_FRAME_SKIP = 1;
    static final int N_BATCH = -1;

    default void train(int batchSize,
                       long framesLimit,
                       ExperienceReplayBuffer replayBuffer,
                       Loss lossFunc) {
        train(batchSize, framesLimit, DEFAULT_FRAME_SKIP, replayBuffer, lossFunc);
    }

    void train(int batchSize,
               long framesLimit,
               int framesSkip,
               ExperienceReplayBuffer replayBuffer,
               Loss lossFunc);

    /**
     * @return true if the agent is in evaluation mode
     */
    boolean test();

    int episodes();

    /**
     * @return the latest metadata returned by the environment step or reset
     */
    Map<Object, Object> lastInfo();

    ActionSpaceType.ActionResult selectAction(NDArray state);

    float reduceEpsilon(float epsilon);

    void save(Path modelPath, String newModelName);

    default double run() {
        return run(1, true).getLast();
    }

    default double run(boolean render) {
        return run(1, render).getLast();
    }

    /**
     *
     * @return totalRewardPerTry
     */
    List<Double> run(int maxTries, boolean render);
}
