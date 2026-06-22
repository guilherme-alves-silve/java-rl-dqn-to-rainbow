package br.com.guialves.rflr.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.util.ProgressBar;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.DJLUtils;
import br.com.guialves.rflr.utils.Experience;
import br.com.guialves.rflr.utils.ExperienceReplayBuffer;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.DISCRETE;

public class AgentDQN {

    private int episodes;
    private int totalFrames;
    private float epsilon;
    private final int updateQTargetAtTimeN;
    private final float minEpsilon;
    private final float epsilonDecay;
    private final float gamma;
    private final IEnv env;
    private final Shape input;
    private final Shape output;
    private final IDeepQNetwork onlineNet;
    private final IDeepQNetwork targetNet;

    public AgentDQN(int episodes, int totalFrames,  float epsilon,
                    int updateQTargetAtTimeN, float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Shape input, Shape output,
                    Supplier<IDeepQNetwork> networkFactory) {
        this.episodes = episodes;
        this.totalFrames = totalFrames;
        this.epsilon = epsilon;
        this.updateQTargetAtTimeN = updateQTargetAtTimeN;
        this.minEpsilon = minEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.gamma = gamma;
        this.env = env;
        this.input = input;
        this.output = output;
        this.onlineNet = networkFactory.get();
        this.targetNet = onlineNet.clone();
    }

    public void train(int batchSize, long framesLimit, ExperienceReplayBuffer replayBuffer) {
        train(batchSize, framesLimit, 1, replayBuffer);
    }

    public void train(int batchSize, long framesLimit, int framesSkip, ExperienceReplayBuffer replayBuffer) {

        var pg = new ProgressBar("training DQN...", framesLimit);
        int frames = 0;
        while (frames < framesLimit) {
            var stateAndInfoMap = env.reset();
            var state = stateAndInfoMap.getKey();
            while (frames < framesLimit) {

                var action = greedyActionSelect(state);

                var stepResult = env.step(action);
                var nextState = stepResult.state();
                var exp = new Experience(state, action, stepResult.reward(), nextState, stepResult.done());

                replayBuffer.store(exp);

                trainQOnline(batchSize, replayBuffer);

                updateTargetNetworkAtN(frames);

                if (stepResult.done()) {
                    ++episodes;
                    break;
                }

                state = nextState;
                epsilon = reduceEpsilon(epsilon);
                frames += framesSkip;
            }
            pg.update(frames);
        }
        pg.end();
    }

    private void trainQOnline(int batchSize, ExperienceReplayBuffer replayBuffer) {
        if (replayBuffer.size() < batchSize) return;

        replayBuffer.sample(batchSize);
    }

    private void updateTargetNetworkAtN(int frames) {
        if (frames % updateQTargetAtTimeN == 0) {
            DJLUtils.copy(onlineNet.getBlock(), targetNet.getBlock());
        }
    }

    public ActionSpaceType.ActionResult greedyActionSelect(NDArray state) {
        var rand = ThreadLocalRandom.current().nextFloat();
        if (rand < epsilon) {
            return env.actionSpaceSample();
        }

        var output = onlineNet.forward(state);
        output.stopGradient();
        return DISCRETE.get(output.argMax().getInt(0));
    }

    public float reduceEpsilon(float epsilon) {
        return Math.max(minEpsilon, epsilon * epsilonDecay);
    }
}
