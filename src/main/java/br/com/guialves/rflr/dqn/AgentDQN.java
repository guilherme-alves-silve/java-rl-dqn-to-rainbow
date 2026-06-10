package br.com.guialves.rflr.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.DISCRETE;

public class AgentDQN {

    private int frames;
    private int episodes;
    private int totalFrames;
    private final float epsilon;
    private final float minEpsilon;
    private final float epsilonDecay;
    private final float gamma;
    private final IEnv env;
    private final Shape input;
    private final Shape output;
    private final IDeepQNetwork onlineNet;
    private final IDeepQNetwork targetNet;

    public AgentDQN(int frames, int episodes, int totalFrames,
                    float epsilon, float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Shape input, Shape output,
                    Supplier<IDeepQNetwork> networkFactory) {
        this.frames = frames;
        this.episodes = episodes;
        this.totalFrames = totalFrames;
        this.epsilon = epsilon;
        this.minEpsilon = minEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.gamma = gamma;
        this.env = env;
        this.input = input;
        this.output = output;
        this.onlineNet = networkFactory.get();
        this.targetNet = onlineNet.clone();
    }

    public void train() {
        var stateAndInfoMap = env.reset();
        while (true) {

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
