package br.com.guialves.rflr.algorithms;

import ai.djl.ndarray.NDArray;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.buffer.IExperience;
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
public abstract class AbstractAgent<T extends IExperience> implements IAgent<T> {

    protected final ActionSpaceType actionSpaceType;

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

    public AbstractAgent(float epsilon, int updateQTargetAtTimeN,
                         float minEpsilon, float epsilonDecay,
                         float gamma, IEnv env, Optimizer optimizer,
                         Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
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
        DJLUtils.freeze(this.targetNet.getBlock());
        this.optimizer = optimizer;
        this.actionSpaceType = env.actionSpaceType();
    }

    @Override
    public void train(int batchSize,
                      long framesLimit,
                      int framesSkip,
                      IReplayBuffer<T> replayBuffer,
                      Loss lossFunc) {

        requireNonNull(replayBuffer, "replayBuffer cannot be null!");
        requireNonNull(lossFunc, "loss cannot be null!");
        plotTrackers.explainTrackersMessage();

        this.test = false;
        this.episodes = 0;

        var parent = env.manager();
        @Cleanup var pbar = ProgressBar.builder()
                .setTaskName("training " + getClass().getSimpleName())
                .setInitialMax(framesLimit)
                .setMaxRenderedLength(200)
                .build();
        int frames = 0;
        do {
            var stateAndInfoMap = env.reset();
            @Cleanup var state = stateAndInfoMap.state();
            var episodeRewards = new ArrayList<>();
            float episodeLossSum = 0f;
            int episodeSteps = 0;

            do {
                @Cleanup var sub = env.newSubManager();
                var action = selectAction(state);

                var stepResult = env.step(action, sub);
                var reward = stepResult.reward();
                var nextState = stepResult.state();
                var done = stepResult.done();
                var info = stepResult.info();
                if (!info.isEmpty()) lastInfo = info;
                var exp = newExperience(state.duplicate(), action, reward,
                        nextState.duplicate(), done);
                replayBuffer.store(exp);

                var lossItem = trainOnline(batchSize, replayBuffer, lossFunc);
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

                state = transfer(parent, state, nextState);

                pbar.stepTo(frames);
                plotTrackers.setTrackersMessage(pbar, frames, replayBuffer.size(), parent);
                templateExtraProcessing(frames, framesLimit);
                epsilon = reduceEpsilon(epsilon);
            } while (frames < framesLimit);
        } while ((frames < framesLimit));
        plotTrackers.setTrackersMessage(pbar, frames, replayBuffer.size(), parent);

        debugDump(env.manager());

        if (getBoolProp("agent.showAllMetrics")) {
            plotTrackers.showAllMetrics();
        }
    }

    /**
     * Follow the design pattern template method
     */
    protected void templateExtraProcessing(int frames, long frameLimit) {
        // Empty, it's a template method
    }

    protected abstract T newExperience(NDArray state,
                                       ActionSpaceType.ActionResult action,
                                       double reward,
                                       NDArray nextState,
                                       boolean done);

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
                                         IReplayBuffer<T> replayBuffer,
                                         Loss lossFunc);

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

        var parent = env.manager();
        var totalRewardPerTry = new ArrayList<Double>(maxTries);
        @Cleanup var envRender = new EnvRenderWindow();
        for (int tries = 0; tries < maxTries; ++tries) {
            @Cleanup var sub = env.newSubManager();
            if (!(env.reset(sub) instanceof EnvResetResult(var state, var _)))
                throw new IllegalStateException();

            double totalReward = 0d;
            boolean done;
            do {
                if (render) envRender.displayAndWait(env.render());
                @Cleanup var action = selectAction(state);
                var stepResult = env.step(action, sub);
                var reward = stepResult.reward();
                var nextState = stepResult.state();
                done = stepResult.done();
                state = transfer(parent, state, nextState);
                totalReward += reward;
            } while (!done);

            totalRewardPerTry.add(totalReward);
        }

        return totalRewardPerTry;
    }
}
