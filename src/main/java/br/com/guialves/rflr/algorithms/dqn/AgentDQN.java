package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.OptimizerUtils;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.utils.DJLUtils;
import br.com.guialves.rflr.utils.Experience;
import br.com.guialves.rflr.utils.ExperienceReplayBuffer;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.extern.slf4j.Slf4j;
import me.tongfei.progressbar.ProgressBar;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.gymnasium4j.EngineUtils.gradCol;
import static java.util.Objects.requireNonNull;

@Slf4j
public class AgentDQN {

    public static final int DEFAULT_FRAME_SKIP = 1;
    private final int[] the2ndAxis = new int[] {1};
    private final ActionSpaceType actionSpaceType;

    private boolean test;
    private int episodes;
    private float epsilon;
    private Map<Object, Object> lastInfo;

    private final int updateQTargetAtTimeN;
    private final float minEpsilon;
    private final float epsilonDecay;
    private final float gamma;
    private final IEnv env;
    private final Device device;
    private final Optimizer optimizer;
    private final IDeepQNetwork onlineNet;
    private final IDeepQNetwork targetNet;
    private final PlotTrackers plotTrackers;

    public AgentDQN(float epsilon, int updateQTargetAtTimeN,
                    float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Device device, Optimizer optimizer,
                    Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
        this.test = false;
        this.epsilon = epsilon;
        this.updateQTargetAtTimeN = updateQTargetAtTimeN;
        this.minEpsilon = minEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.gamma = gamma;
        this.env = env;
        this.device = device;
        this.onlineNet = networkFactory.get();
        this.plotTrackers = plotTrackers;
        this.targetNet = onlineNet.clone();
        DJLUtils.freeze(this.targetNet.getBlock());
        this.optimizer = optimizer;
        this.actionSpaceType = env.actionSpaceType();
    }

    public void train(int batchSize,
                      long framesLimit,
                      ExperienceReplayBuffer replayBuffer,
                      Loss lossFunc) {
        train(batchSize, framesLimit, DEFAULT_FRAME_SKIP, replayBuffer, lossFunc);
    }

    public void train(int batchSize,
                      long framesLimit,
                      int framesSkip,
                      ExperienceReplayBuffer replayBuffer,
                      Loss lossFunc) {

        requireNonNull(replayBuffer, "replayBuffer cannot be null!");
        requireNonNull(replayBuffer, "loss cannot be null!");

        this.test = false;
        this.episodes = 0;

        var pbar = ProgressBar.builder()
                .setTaskName("training DQN")
                .setInitialMax(framesLimit)
                .setMaxRenderedLength(100)
                .build();
        try (pbar) {
            int frames = 0;
            while (frames < framesLimit) {
                var stateAndInfoMap = env.reset();
                var state = stateAndInfoMap.state();
                var episodeRewards = new ArrayList<>();
                float episodeLossSum = 0f;
                int episodeSteps = 0;

                while (frames < framesLimit) {

                    var action = selectAction(state);

                    var stepResult = env.step(action);
                    var reward = stepResult.reward();
                    var nextState = stepResult.state();
                    var done = stepResult.done();
                    var info = stepResult.info();
                    if (!info.isEmpty()) lastInfo = info;
                    var exp = new Experience(state.duplicate(), action, reward,
                            nextState.duplicate(), done);
                    replayBuffer.store(exp);

                    var lossItem = trainQOnline(batchSize, replayBuffer, lossFunc);
                    if (!Float.isNaN(lossItem)) {
                        episodeLossSum += lossItem;
                        ++episodeSteps;
                    }

                    frames += framesSkip;
                    updateTargetNetworkAtN(frames);

                    episodeRewards.add(reward);
                    if (done) {
                        ++episodes;
                        float avgLoss = episodeSteps > 0 ? episodeLossSum / episodeSteps : 0;
                        plotTrackers.add(epsilon, episodeRewards, avgLoss);
                        break;
                    }

                    state.close();
                    state = nextState;

                    pbar.stepTo(frames);
                }

                epsilon = reduceEpsilon(epsilon);
            }
            plotTrackers.setTrackersMessage(pbar, frames);
        }

        plotTrackers.showAllMetrics();
    }

    /**
     * @return true if the agent is in evaluation mode
     */
    public boolean test() {
        return test;
    }

    public int episodes() {
        return episodes;
    }

    /**
     * @return the latest metadata returned by the environment step or reset
     */
    public Map<Object, Object> lastInfo() {
        return lastInfo;
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

            // max q_target(s', a')
            var maxNextQValue = targetNet.forward(nextStates)
                    .max(the2ndAxis, true);
            // gamma * max q_target(s', a')
            var discountNextQValue = maxNextQValue.mul(gamma);
            // (1 - done)
            var mask = dones.neg().add(1);
            // y = r + gamma * max q_target(s', a') * (1 - done)
            var targetQValue = rewards.add(discountNextQValue.mul(mask))
                    .stopGradient();

            float lossItem;
            try (var gc = gradCol()) {
                // y_hat = q_online(s, a)
                var qValue = onlineNet.forward(states).gather(actions, 1);
                var lossVal = lossFunc.evaluate(new NDList(qValue), new NDList(targetQValue));
                gc.backward(lossVal);
                lossItem = lossVal.stopGradient().mean().getFloat();
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

    public ActionSpaceType.ActionResult selectAction(NDArray state) {
        if (!test) {
            var rand = ThreadLocalRandom.current().nextFloat();
            if (rand < epsilon) {
                return env.actionSpaceSample();
            }
        }

        try(var output = onlineNet.forward(state.expandDims(0)).stopGradient().argMax(1)) {
            return actionSpaceType.get(output.getLong(0));
        }
    }

    public float reduceEpsilon(float epsilon) {
        return Math.max(minEpsilon, epsilon * epsilonDecay);
    }

    public void save(Path modelPath, String newModelName) {
        this.onlineNet.save(modelPath, newModelName);
    }

    public double run() {
        return run(1, true).getLast();
    }

    public double run(boolean render) {
        return run(1, render).getLast();
    }

    /**
     *
     * @return totalRewardPerTry
     */
    public List<Double> run(int maxTries, boolean render) {
        this.test = true;

        var totalRewardPerTry = new ArrayList<Double>(maxTries);
        try (var envRender = new EnvRenderWindow()) {
            for (int tries = 0; tries < maxTries; ++tries) {
                if (!(env.reset() instanceof EnvResetResult(var state, var _)))
                    throw new IllegalStateException();

                double totalReward = 0d;
                boolean done;
                do {
                    if (render) envRender.displayAndWait(env.render());
                    var action = selectAction(state);
                    var stepResult = env.step(action);
                    var reward = stepResult.reward();
                    var nextState = stepResult.state();
                    done = stepResult.done();
                    state = nextState;
                    totalReward += reward;
                } while (!done);

                totalRewardPerTry.add(totalReward);
            }
        }

        return totalRewardPerTry;
    }
}
