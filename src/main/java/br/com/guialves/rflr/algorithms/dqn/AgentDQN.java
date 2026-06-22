package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.OptimizerUtils;
import br.com.guialves.rflr.utils.DJLUtils;
import br.com.guialves.rflr.utils.Experience;
import br.com.guialves.rflr.utils.ExperienceReplayBuffer;
import br.com.guialves.rflr.utils.PlotTrackers;
import me.tongfei.progressbar.ProgressBar;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.gymnasium4j.ActionSpaceType.DISCRETE;
import static br.com.guialves.rflr.gymnasium4j.EngineUtils.gradCol;
import static java.util.Objects.requireNonNull;

public class AgentDQN {

    private final int[] the2ndAxis = new int[] {1};

    private int episodes;
    private int totalFrames;
    private float epsilon;
    private final int updateQTargetAtTimeN;
    private final float learningRate;
    private final float minEpsilon;
    private final float epsilonDecay;
    private final float gamma;
    private final IEnv env;
    private final Shape inputShape;
    private final Shape outputShape;
    private final Optimizer optimizer;
    private final IDeepQNetwork onlineNet;
    private final IDeepQNetwork targetNet;
    private final PlotTrackers plotTrackers;

    public AgentDQN(int episodes, int totalFrames, float epsilon,
                    int updateQTargetAtTimeN, float learningRate, float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Shape inputShape, Shape outputShape,
                    Optimizer optimizer, Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
        this.episodes = episodes;
        this.totalFrames = totalFrames;
        this.epsilon = epsilon;
        this.updateQTargetAtTimeN = updateQTargetAtTimeN;
        this.minEpsilon = minEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.gamma = gamma;
        this.env = env;
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.onlineNet = networkFactory.get();
        this.plotTrackers = plotTrackers;
        this.targetNet = onlineNet.clone();
        this.learningRate = learningRate;
        this.optimizer = optimizer;
    }

    public void train(int batchSize, long framesLimit, ExperienceReplayBuffer replayBuffer, Loss lossFunc) {
        train(batchSize, framesLimit, replayBuffer, lossFunc);
    }

    public void train(int batchSize,
                      long framesLimit,
                      int framesSkip,
                      ExperienceReplayBuffer replayBuffer,
                      Loss lossFunc) {

        requireNonNull(replayBuffer, "replayBuffer cannot be null!");
        requireNonNull(replayBuffer, "loss cannot be null!");

        try (var pg = new ProgressBar("training DQN...", framesLimit)) {
            int frames = 0;
            while (frames < framesLimit) {
                var stateAndInfoMap = env.reset();
                var state = stateAndInfoMap.state();
                var episodeRewards = new ArrayList<>();
                float episodeLossSum = 0f;
                int episodeSteps = 0;

                while (frames < framesLimit) {

                    var action = greedyActionSelect(state);

                    var stepResult = env.step(action);
                    var reward = stepResult.reward();
                    var nextState = stepResult.state();
                    var done = stepResult.done();
                    var exp = new Experience(state, action, reward, nextState, done);

                    replayBuffer.store(exp);

                    var lossItem = trainQOnline(batchSize, replayBuffer, lossFunc);
                    if (!Float.isNaN(lossItem)) {
                        episodeLossSum += lossItem;
                        ++episodeSteps;
                    }

                    updateTargetNetworkAtN(frames);

                    episodeRewards.add(reward);
                    if (done) {
                        ++episodes;
                        float avgLoss = episodeSteps > 0 ? episodeLossSum / episodeSteps : 0;
                        plotTrackers.add(epsilon, episodeRewards, avgLoss);
                        break;
                    }

                    state = nextState;
                    frames += framesSkip;
                }
                epsilon = reduceEpsilon(epsilon);
                pg.stepTo(frames);
                plotTrackers.setTrackersMessage(pg, frames);
            }
        }
    }

    /**
     * Method used to train the Q-online network
     * Observations: In case you want to understand the shape/dimensions of the
     * operations below, you can check the PlaygroundMatrixOperationsTest.
     * Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-scratch.html">DJL Linear Regression from Scratch</a>
     * @param batchSize Number of experiences to sample from the replay buffer for each training step
     * @param replayBuffer Experience replay buffer containing stored transitions (state, action, reward, nextState, done)
     * @param lossFunc Loss function used to compute the difference between current Q-values and target Q-values (e.g., MSE, Huber)
     */
    private float trainQOnline(int batchSize, ExperienceReplayBuffer replayBuffer, Loss lossFunc) {
        if (replayBuffer.size() < batchSize) return Float.NaN;

        try(var samples = replayBuffer.sample(batchSize)) {
            var states = samples.states();
            var actions = samples.actions();
            var rewards = samples.rewards();
            var nextStates = samples.nextStates();
            var dones = samples.dones();

            // y_hat = q_online(s, a)
            var qValue = onlineNet.forward(states);
            // max q_target(s', a')
            var maxNextQValue = targetNet.forward(nextStates)
                    .max(the2ndAxis)
                    .gather(actions.expandDims(1), 1);
            // gamma * max q_target(s', a')
            var discountNextQValue = maxNextQValue.mul(gamma);
            // (1 - done)
            var mask = dones.neg().add(1);
            // y = r + gamma * max q_target(s', a') * (1 - done)
            var targetQValue = rewards.add(discountNextQValue.mul(mask));
            targetQValue = targetQValue.stopGradient();

            float lossItem;
            try (var gc = gradCol()) {
                var lossVal = lossFunc.evaluate(new NDList(qValue), new NDList(targetQValue));
                gc.backward(lossVal);
                lossItem = lossVal.stopGradient().mean().getFloat(0);
            }
            OptimizerUtils.trainStep(onlineNet.getBlock(), optimizer);
            return lossItem;
        }
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
