package br.com.guialves.rflr.algorithms;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.INoisyNetwork;
import br.com.guialves.rflr.djlutils.DJLMemoryLeakChecker;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvResetResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.utils.ModelFileNames;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.SneakyThrows;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import me.tongfei.progressbar.ProgressBar;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLUtils.*;
import static br.com.guialves.rflr.utils.PropUtils.getBoolProp;
import static java.util.Objects.requireNonNull;

@Slf4j
@Accessors(fluent = true)
public abstract class AbstractAgent implements IAgent {

    /**
     * The minimum interval between two updates of the progress bar on the console.
     */
    private static final long PROGRESS_UPDATE_INTERVAL_NANOS = 200_000_000L;
    protected final ActionSpaceType actionSpaceType;
    protected final NDManager parent;
    protected final DJLMemoryLeakChecker memLeakChecker;

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

    private long lastProgressUpdate = 0;

    public AbstractAgent(float epsilon,
                         int updateQTargetAtTimeN,
                         float minEpsilon,
                         float epsilonDecay,
                         float gamma,
                         IEnv env,
                         Optimizer optimizer,
                         NDManager parent,
                         Supplier<IDeepQNetwork> networkFactory,
                         PlotTrackers plotTrackers,
                         boolean debugMemoryLeak) {
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
        freeze(this.targetNet.getBlock());
        this.optimizer = optimizer;
        this.parent = parent;
        parent.cap();
        this.actionSpaceType = env.actionSpaceType();
        this.memLeakChecker = new DJLMemoryLeakChecker(debugMemoryLeak, parent);
    }

    @Override
    public void train(int batchSize,
                      long framesLimit,
                      int framesSkip,
                      IReplayBuffer replayBuffer,
                      Loss lossFunc) {

        requireNonNull(replayBuffer, "replayBuffer cannot be null!");
        requireNonNull(lossFunc, "loss cannot be null!");

        if (env.vectorized()) {
            trainVectorized(batchSize, framesLimit, framesSkip, replayBuffer, lossFunc);
            return;
        }

        plotTrackers.explainTrackersMessage();

        this.test = false;
        this.episodes = 0;

        @Cleanup var pbar = ProgressBar.builder()
                .setTaskName("training " + getClass().getSimpleName())
                .setInitialMax(framesLimit)
                .setMaxRenderedLength(200)
                .build();
        int frames = 0;
        long steps = 0;
        int snapshotInterval = memLeakChecker.snapshotInterval(framesLimit, framesSkip);
        do {
            @Cleanup var parentPerEpisode = subMgr(parent, "parent-episode");
            var stateAndInfoMap = env.reset(parentPerEpisode);
            @Cleanup var state = stateAndInfoMap.state();
            var episodeRewards = new ArrayList<Double>();
            float episodeLossSum = 0f;
            int episodeSteps = 0;

            do {
                @Cleanup var sub = subMgr(parentPerEpisode, "sub-frame");
                @Cleanup var action = selectAction(state);

                var stepResult = env.step(action, sub);
                var reward = stepResult.reward();
                var nextState = stepResult.state();
                var term = stepResult.term();
                var trunc = stepResult.trunc();
                var done = stepResult.done();
                var info = stepResult.info();
                if (!info.isEmpty()) lastInfo = info;
                var exp = Experience.copy(state, action, reward, nextState, term, trunc);
                replayBuffer.store(exp);

                float lossItem = trainOnline(batchSize, replayBuffer, lossFunc, sub);
                if (trained(lossItem)) {
                    episodeLossSum += lossItem;
                    ++episodeSteps;
                }

                frames += framesSkip;
                ++steps;
                updateTargetNetworkAtN(frames);

                episodeRewards.add(reward);
                if (done) {
                    ++episodes;
                    float avgLoss = episodeSteps > 0 ? episodeLossSum / episodeSteps : 0;
                    plotTrackers.add(epsilon, episodeRewards, avgLoss);
                    memLeakChecker.offerMemorySnapshot(steps, snapshotInterval);
                    release(state, nextState);
                    break;
                }

                state = transfer(parentPerEpisode, state, nextState);
                memLeakChecker.offerMemorySnapshot(steps, snapshotInterval);
                updateProgressBar(pbar, frames, replayBuffer.size(), parentPerEpisode);
                templateExtraProcessing(frames, framesLimit);
                epsilon = reduceEpsilon(epsilon);
            } while (frames < framesLimit);
        } while ((frames < framesLimit));

        memLeakChecker.offerMemorySnapshot();

        pbar.stepTo(frames);
        plotTrackers.updateTrackersMessage(pbar, frames, replayBuffer.size(), parent);

        if (getBoolProp("agent.showAllMetrics")) {
            plotTrackers.showAllMetrics();
        }
    }

    /**
     * Updates the progress bar and the trackers of the console.
     *
     * <p>The update is throttled: writing the progress bar on the console costs a few
     * milliseconds, and doing it on every step of a vectorized training (where a step is a few
     * milliseconds itself) is a significant part of the training time.</p>
     *
     * @param pbar             the progress bar
     * @param frames           the number of frames played so far
     * @param bufferSize       the number of experiences in the replay buffer
     * @param parentPerEpisode the parent manager of the episode, when there is one
     */
    private void updateProgressBar(ProgressBar pbar,
                                   int frames,
                                   int bufferSize,
                                   NDManager parentPerEpisode) {
        long now = System.nanoTime();
        if (now - lastProgressUpdate < PROGRESS_UPDATE_INTERVAL_NANOS) return;
        lastProgressUpdate = now;
        pbar.stepTo(frames);
        plotTrackers.updateTrackersMessage(pbar, frames, bufferSize, parent, parentPerEpisode);
    }

    private boolean trained(float lossItem) {
        return !Float.isNaN(lossItem);
    }

    /**
     * Vectorized training loop: all the environments are stepped in lockstep and
     * a single gradient step is performed per vectorized step.
     *
     * <p>The observation batch has the shape {@code (numEnvs, ...)} and is fed to
     * the online network in a single batched forward pass, which is the whole
     * point of the vectorization: the GPU sees {@code numEnvs} samples at once
     * instead of one.</p>
     *
     * <p>The frames counter follows the <b>emulator frames</b>, so one vectorized
     * step advances {@code framesSkip * numEnvs} frames.</p>
     *
     * @param batchSize     the mini-batch size
     * @param framesLimit   the training budget, in emulator frames
     * @param framesSkip    the number of emulator frames consumed by one env step
     * @param replayBuffer  the buffer that receives the transitions
     * @param lossFunc      the loss function
     */
    private void trainVectorized(int batchSize,
                                 long framesLimit,
                                 int framesSkip,
                                 IReplayBuffer replayBuffer,
                                 Loss lossFunc) {
        plotTrackers.explainTrackersMessage();

        this.test = false;
        this.episodes = 0;

        @Cleanup var pbar = ProgressBar.builder()
                .setTaskName("training " + getClass().getSimpleName())
                .setInitialMax(framesLimit)
                .setMaxRenderedLength(200)
                .build();

        @Cleanup var parentVec = subMgr(parent, "parent-vectorized");
        var stateAndInfo = env.reset(parentVec);
        @Cleanup var states = stateAndInfo.state();
        // (Envs, Channels, Height, Width)
        int numberOfEnvs = (int) states.getShape().get(0);
        long realFramesSkip = (long) framesSkip * numberOfEnvs;

        var episodeRewards = new ArrayList<Double>(numberOfEnvs);
        IntStream.rangeClosed(0, numberOfEnvs).forEach(_ -> episodeRewards.add(0d));

        float lossSum = 0f;
        int lossSteps = 0;

        long frames = 0;
        long steps = 0;
        int snapshotInterval = memLeakChecker.snapshotInterval(framesLimit, realFramesSkip);
        while (frames < framesLimit) {
            @Cleanup var sub = subMgr(parentVec, "sub-vectorized-step");
            states.attach(sub);
            @Cleanup var actions = selectActions(states);
            @Cleanup var vecResult = env.stepVec(actions, sub);

            for (int idxEnv = 0; idxEnv < numberOfEnvs; ++idxEnv) {
                var step = vecResult.step(idxEnv);
                @Cleanup var state = states.get(idxEnv);
                var reward = step.reward();
                var done = step.done();
                var nextState = step.state();
                var exp = Experience.copy(idxEnv, state, actions.toDiscrete(idxEnv), step.reward(),
                        nextState, step.term(), step.trunc());
                replayBuffer.store(exp);

                episodeRewards.set(idxEnv, episodeRewards.get(idxEnv) + reward);
                if (done) {
                    ++episodes;
                    float avgLoss = lossSteps > 0? lossSum / lossSteps : 0;
                    plotTrackers.add(epsilon, List.of(episodeRewards.get(idxEnv)), avgLoss);
                    episodeRewards.set(idxEnv, 0d);
                }
            }

            float lossItem = trainOnline(batchSize, replayBuffer, lossFunc, sub);
            if (trained(lossItem)) {
                lossSum += lossItem;
                ++lossSteps;
            }

            var nextStates = vecResult.states();
            states = transferDetached(states, nextStates);

            frames += realFramesSkip;
            ++steps;
            epsilon = reduceEpsilon(epsilon, numberOfEnvs);
            updateTargetNetworkAtN((int) frames);

            memLeakChecker.offerMemorySnapshot(steps, snapshotInterval);
            pbar.stepTo(frames);
            plotTrackers.updateTrackersMessage(pbar, (int) frames, replayBuffer.size(), parent, parentVec);
            templateExtraProcessing((int) frames, framesLimit);
        }

        memLeakChecker.offerMemorySnapshot();

        plotTrackers.updateTrackersMessage(pbar, (int) frames, replayBuffer.size(), parent);

        if (getBoolProp("agent.showAllMetrics")) {
            plotTrackers.showAllMetrics();
        }
    }

    /**
     * Selects one action per environment with a single batched forward pass.
     *
     * @param states the batched observation, shape {@code (numEnvs, ...)}
     * @return the greedy action of each environment
     */
    protected ActionSpaceType.ActionResult selectActions(NDArray states) {
        if (onlineNet instanceof INoisyNetwork noisyNetwork) {
            noisyNetwork.resetNoise();
        }

        if (!test) {
            var rand = ThreadLocalRandom.current().nextFloat();
            if (rand < epsilon) {
                return env.actionSpaceSample();
            }
        }

        long[] actions = onlineNet.forwardLongs(states,
                qValue -> qValue.stopGradient().argMax(AXIS_1));
        return actionSpaceType.get(actions);
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
        if (null == lastInfo) return Collections.emptyMap();
        return lastInfo;
    }

    protected abstract float trainOnline(int batchSize,
                                         IReplayBuffer replayBuffer,
                                         Loss lossFunc,
                                         NDManager sub);

    protected void updateTargetNetworkAtN(int frames) {
        if (frames % updateQTargetAtTimeN == 0) {
            copy(onlineNet.getBlock(), targetNet.getBlock());
            freeze(targetNet.getBlock());
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
        long action = onlineNet.forwardLong(oneBatchState,
                qValue -> qValue.stopGradient().argMax(AXIS_1));
        return actionSpaceType.get(action);
    }

    @Override
    public float reduceEpsilon(float epsilon) {
        return Math.max(minEpsilon, epsilon - epsilonDecay);
    }

    /**
     * Reduces the exploration rate taking into account that one vectorized step
     * advances {@code steps} environment steps.
     *
     * @param epsilon the current exploration rate
     * @param steps   the number of environment steps performed
     * @return the new exploration rate
     */
    protected float reduceEpsilon(float epsilon, int steps) {
        return Math.max(minEpsilon, epsilon - epsilonDecay * steps);
    }

    @Override
    public void save(Path modelPath, String newModelName) {
        ModelFileNames.ensureParentDirectory(modelPath, newModelName);
        this.onlineNet.save(modelPath, newModelName);
    }

    @Override
    public List<Double> run(int maxTries, boolean render) {
        this.test = true;
        this.onlineNet.eval();

        var totalRewardPerTry = new ArrayList<Double>(maxTries);
        for (int tries = 0; tries < maxTries; ++tries) {
            @Cleanup var sub = subMgr(parent, "run-episode");
            var totalReward = env.vectorized()?
                    runVectorizedEpisode(sub, render) :
                    runSingleEpisode(sub, render);
            totalRewardPerTry.add(totalReward);
        }

        return totalRewardPerTry;
    }

    /**
     * Runs one episode of a single environment.
     *
     * @param sub    the manager of the episode
     * @param render whether the episode must be rendered
     * @return the reward of the episode
     */
    private double runSingleEpisode(NDManager sub, boolean render) {
        @Cleanup var envRender = new EnvRenderWindow();
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

        return totalReward;
    }

    /**
     * Runs one episode of a vectorized environment: every environment advances at the same time,
     * with one batched forward pass, and the reward of each environment is accumulated until it
     * is done.
     *
     * <p>The environment auto-resets, so the reward of an environment that is already done is not
     * accumulated anymore.</p>
     *
     * @param sub    the manager of the episode
     * @param render whether the episode must be rendered
     * @return the mean reward of the episode of the environments
     */
    private double runVectorizedEpisode(NDManager sub, boolean render) {
        if (!(env.reset(sub) instanceof EnvResetResult(var state, var _)))
            throw new IllegalStateException("Must be of type EnvResetResult!");

        int numberOfEnvs = (int) state.getShape().get(0);
        @Cleanup var envRender = new EnvRenderWindow(numberOfEnvs);
        var totalReward = new double[numberOfEnvs];
        var done = new boolean[numberOfEnvs];
        int pending = numberOfEnvs;

        while (pending > 0) {
            if (render) envRender.displayAndWait(env.renderVec());
            @Cleanup var actions = selectActions(state);
            @Cleanup var stepResults = env.stepVec(actions, sub);
            for (int idxEnv = 0; idxEnv < numberOfEnvs; ++idxEnv) {
                if (done[idxEnv]) continue;
                var step = stepResults.step(idxEnv);
                totalReward[idxEnv] += step.reward();
                if (step.done()) {
                    done[idxEnv] = true;
                    --pending;
                }
            }
            state = transfer(sub, state, stepResults.states());
        }

        return Arrays.stream(totalReward)
                .average()
                .orElse(0d);
    }

    @Override
    public List<ManagerNode> getManagerNodes() {
        return memLeakChecker.getManagerNodes();
    }

    @Override
    @SneakyThrows
    public void close() {
        onlineNet.close();
        targetNet.close();
    }
}
