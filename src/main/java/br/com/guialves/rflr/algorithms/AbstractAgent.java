package br.com.guialves.rflr.algorithms;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLUtils;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;
import me.tongfei.progressbar.ProgressBar;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static java.util.Objects.requireNonNull;

@Slf4j
public abstract class AbstractAgent implements IAgent {

    protected final ActionSpaceType actionSpaceType;
    protected final NDManager parent;

    protected boolean test;
    protected int episodes;
    protected float epsilon;
    protected Map<Object, Object> lastInfo;

    protected final int updateQTargetAtTimeN;
    protected final float minEpsilon;
    protected final float epsilonDecay;
    protected final float gamma;
    protected final IEnv env;
    protected final Optimizer optimizer;
    protected final IDeepQNetwork onlineNet;
    protected final IDeepQNetwork targetNet;
    protected final PlotTrackers plotTrackers;

    public AbstractAgent(float epsilon,
                         int updateQTargetAtTimeN,
                         float minEpsilon,
                         float epsilonDecay,
                         float gamma,
                         IEnv env,
                         Optimizer optimizer,
                         NDManager parent,
                         Supplier<IDeepQNetwork> networkFactory,
                         PlotTrackers plotTrackers) {
        log.info("Creating {}", getClass().getSimpleName());
        this.test = false;
        this.epsilon = epsilon;
        this.updateQTargetAtTimeN = updateQTargetAtTimeN;
        this.minEpsilon = minEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.gamma = gamma;
        this.env = env;
        this.onlineNet = networkFactory.get();
        this.plotTrackers = plotTrackers;
        this.targetNet = onlineNet.clone();
        this.targetNet.eval();
        DJLUtils.freeze(this.targetNet.getBlock());
        this.optimizer = optimizer;
        this.parent = parent;
        parent.cap();
        this.actionSpaceType = env.actionSpaceType();
    }

    @Override
    public void train(int batchSize,
                      long framesLimit,
                      int framesSkip,
                      IReplayBuffer replayBuffer,
                      Loss lossFunc) {

        requireNonNull(replayBuffer, "replayBuffer cannot be null!");
        requireNonNull(lossFunc, "loss cannot be null!");
        plotTrackers.explainTrackersMessage();

        this.test = false;
        this.episodes = 0;

        @Cleanup var pbar = ProgressBar.builder()
                .setTaskName("training " + getClass().getSimpleName())
                .setInitialMax(framesLimit)
                .setMaxRenderedLength(200)
                .build();
        int frames = 0;
        do {
            @Cleanup var parentPerEpisode = parent.newSubManager();
            parentPerEpisode.setName("parentPerEpisode-" + parentPerEpisode.getName());

            var stateAndInfoMap = env.reset(parentPerEpisode);
            @Cleanup var state = stateAndInfoMap.state();
            var episodeRewards = new ArrayList<Double>();
            float episodeLossSum = 0f;
            int episodeSteps = 0;

            do {
                @Cleanup var sub = parentPerEpisode.newSubManager();
                sub.setName("innersub-" + sub.getName());
                @Cleanup var action = selectAction(state);

                var stepResult = env.step(action, sub);
                var reward = stepResult.reward();
                var nextState = stepResult.state();
                var done = stepResult.done();
                var info = stepResult.info();
                if (!info.isEmpty()) lastInfo = info;
                var exp = new Experience(state.duplicate(), action.duplicate(), reward,
                        nextState.duplicate(), done);
                replayBuffer.store(exp);

                float lossItem = trainOnline(batchSize, replayBuffer, lossFunc, sub);
                if (trained(lossItem)) {
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
                    close(state, nextState);
                    break;
                }

                state = transfer(parentPerEpisode, state, nextState);

                pbar.stepTo(frames);
                plotTrackers.setTrackersMessage(pbar, frames, replayBuffer.size(), sub);
                templateExtraProcessing(frames, framesLimit);
                epsilon = reduceEpsilon(epsilon);
            } while (frames < framesLimit);
        } while ((frames < framesLimit));
        plotTrackers.setTrackersMessage(pbar, frames, replayBuffer.size(), parent);

        debugDump(parent);

        if (getBoolProp("agent.showAllMetrics")) {
            plotTrackers.showAllMetrics();
        }
    }

    private boolean trained(float lossItem) {
        return !Float.isNaN(lossItem);
    }

    /**
     * Follow the design pattern template method
     */
    protected void templateExtraProcessing(int frames, long frameLimit) {
        // Empty, it's a template method
    }

    /**
     * @return true if the agent is in evaluation mode
     */
    @Override
    public boolean test() {
        return test;
    }

    @Override
    public int episodes() {
        return episodes;
    }

    /**
     * @return the latest metadata returned by the environment step or reset
     */
    @Override
    public Map<Object, Object> lastInfo() {
        return lastInfo;
    }

    protected abstract float trainOnline(int batchSize,
                                         IReplayBuffer replayBuffer,
                                         Loss lossFunc,
                                         NDManager sub);

    protected void updateTargetNetworkAtN(int frames) {
        if (frames % updateQTargetAtTimeN == 0) {
            DJLUtils.copy(onlineNet.getBlock(), targetNet.getBlock());
            DJLUtils.freeze(targetNet.getBlock());
        }
    }

    @Override
    public ActionSpaceType.ActionResult selectAction(NDArray state) {
        if (!test) {
            var rand = ThreadLocalRandom.current().nextFloat();
            if (rand < epsilon) {
                return env.actionSpaceSample();
            }
        }

        @Cleanup var oneBatchState = state.expandDims(0);
        @Cleanup var output = onlineNet.forward(oneBatchState,
                qValue -> qValue.stopGradient().argMax(1));
        return actionSpaceType.get(output.getLong(0));
    }

    @Override
    public float reduceEpsilon(float epsilon) {
        return Math.max(minEpsilon, epsilon - epsilonDecay);
    }

    @Override
    public void save(Path modelPath, String newModelName) {
        this.onlineNet.save(modelPath, newModelName);
    }

    @Override
    public List<Double> run(int maxTries, boolean render) {
        this.test = true;

        var totalRewardPerTry = new ArrayList<Double>(maxTries);
        @Cleanup var envRender = new EnvRenderWindow();
        for (int tries = 0; tries < maxTries; ++tries) {
            @Cleanup var sub = parent.newSubManager();
            if (!(env.reset(sub) instanceof EnvResetResult(var state, var _)))
                throw new IllegalStateException("Must be of type EnvResetResult!");

            double totalReward = 0d;
            boolean done;
            do {
                if (render) envRender.displayAndWait(env.render());
                @Cleanup var action = selectAction(state);
                var stepResult = env.step(action, sub);
                var reward = stepResult.reward();
                var nextState = stepResult.state();
                done = stepResult.done();
                state = transfer(sub, state, nextState);
                totalReward += reward;
            } while (!done);

            totalRewardPerTry.add(totalReward);
        }

        return totalRewardPerTry;
    }
}
