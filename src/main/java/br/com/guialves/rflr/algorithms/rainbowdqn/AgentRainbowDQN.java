package br.com.guialves.rflr.algorithms.rainbowdqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.NStepPrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.RainbowQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer.MIN_PRIORITY;
import static br.com.guialves.rflr.djlutils.DJLLoss.rawBackwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.*;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStepClipGradients;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1;
import static br.com.guialves.rflr.djlutils.DJLUtils.N_BATCH;

/**
 * Rainbow DQN — the union of all seven improvements from Hessel et al. (2017):
 * <ol>
 *   <li>DQN baseline (target network + replay buffer)</li>
 *   <li>Double DQN (action selection on the online net)</li>
 *   <li>Prioritized Experience Replay (proportional sampling, importance-sampling weights)</li>
 *   <li>Dueling network (value + advantage streams)</li>
 *   <li>N-Step Learning returns ({@code n}-step)</li>
 *   <li>Noisy networks (parametric noise for exploration, no ε-greedy)</li>
 *   <li>C51 distributional (categorical Bellman projection + cross-entropy loss)</li>
 * </ol>
 *
 * <p>The buffer must be a {@link PrioritizedReplayBuffer} or {@link NStepPrioritizedReplayBuffer};
 * the loss must be {@link CategoricalCrossEntropyPERLoss}; the networks must be {@link RainbowQNetworkMLP}.
 */
@Slf4j
public class AgentRainbowDQN extends AbstractAgent {

    private static final float CLIP_GRAD_THRESHOLD = 10.0f;

    private final RainbowQNetworkMLP onlineRainbowNet;
    private final RainbowQNetworkMLP targetRainbowNet;
    private final NDManager subManager;
    private final NDArray atomsBroadcaster;
    private final float initialBeta;
    private float beta;

    public AgentRainbowDQN(float epsilon,
                           int updateQTargetAtTimeN,
                           float minEpsilon,
                           float epsilonDecay,
                           float gamma,
                           float beta,
                           IEnv env,
                           Optimizer optimizer,
                           NDManager parent,
                           Supplier<IDeepQNetwork> networkFactory,
                           PlotTrackers plotTrackers,
                           boolean debugMemoryLeak) {
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, parent,
                networkFactory, plotTrackers, debugMemoryLeak);
        if (!(onlineNet instanceof RainbowQNetworkMLP)) {
            throw new IllegalArgumentException("Invalid network type! Must be of type NoisyDuelingQNetworkMLP!");
        }

        this.onlineRainbowNet = (RainbowQNetworkMLP) onlineNet;
        this.targetRainbowNet = (RainbowQNetworkMLP) targetNet;
        this.subManager = subMgr(parent, "sub-atoms-broadcast");
        this.atomsBroadcaster = this.targetRainbowNet.newAtomsBroadcaster(subManager);
        this.beta = this.initialBeta = beta;
    }

    /**
     * Method used to train the Q-online network
     * Observations: In case you want to understand the shape/dimensions of the
     * operations below, you can check the PlaygroundMatrixOperationsTest.
     * Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-scratch.html">DJL Linear Regression from Scratch</a>
     * @param batchSize Number of experiences to sample from the replay buffer for each training step
     * @param ireplayBuffer Experience replay buffer containing stored transitions (state, action, reward, nextState, done)
     * @param lossFunc Loss function used to compute the difference between current Q-values and target Q-values (e.g., MSE, Huber)
     */
    @Override
    protected float trainOnline(int batchSize, IReplayBuffer ireplayBuffer, Loss lossFunc, NDManager sub) {
        if (!ireplayBuffer.enough(batchSize)) return Float.NaN;
        if (!(ireplayBuffer instanceof NStepPrioritizedReplayBuffer replayBuffer)) {
            throw new IllegalArgumentException("You must pass PrioritizedReplayBuffer!");
        }
        if (!(lossFunc instanceof CategoricalCrossEntropyPERLoss catLossFunc)) {
            throw new IllegalArgumentException("You must pass CategoricalCrossEntropyPERLoss!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize, beta);

        // reset first time eps' for DDQN
        onlineRainbowNet.resetNoise();
        // a* = arg max q_online(s', a')
        var nextStates = samples.nextStates();
        @Cleanup var action = onlineRainbowNet.forward(nextStates,
                onlineNextQValues -> onlineNextQValues.stopGradient().argMax(AXIS_1)
                        // (batch, 1, 1)
                        .reshape(N_BATCH, 1, 1)
                        // (batch, 1, atoms)
                        .mul(atomsBroadcaster));

        @Cleanup var targetQValue = targetRainbowNet.forwardDist(nextStates, probNextDist -> {
            float gammaNBootstrap = (float) Math.pow(gamma, replayBuffer.nStep());
            // q_target(s', a*) - DDQN
            var bestNextProbDist = probNextDist.gather(action, AXIS_1);
            // Bellman Projection - mi (with n-step gammaNBootstrap instead of just gamma)
            return targetRainbowNet.projectBellman(bestNextProbDist, samples.rewards(), samples.dones(), gammaNBootstrap);
        });

        // reset second time eps' for DDQN now to compute TD-Error
        onlineRainbowNet.resetNoise();
        catLossFunc.normISWeights(samples.weights());
        var losses = rawBackwardLoss(sub, catLossFunc, targetQValue, array -> {
            var states = array[0];
            var actions = array[1]
                    // (batch, 1, 1)
                    .reshape(N_BATCH, 1, 1)
                    // (batch, 1, atoms)
                    .mul(atomsBroadcaster);
            // ln(p(s, a, theta))
            return onlineRainbowNet.forwardLogDist(states, logProbDist -> logProbDist.gather(actions, AXIS_1));
        }, samples.states(), samples.actions());

        var lossItem = scopedToFloat(NDArray::mean, losses);
        @Cleanup var priorities = scoped(it -> it.abs().add(MIN_PRIORITY), losses);

        replayBuffer.updatePriorities(samples.bufferIndexes(), priorities);
        trainStepClipGradients(onlineRainbowNet.getBlock(), optimizer, CLIP_GRAD_THRESHOLD);
        return lossItem;
    }

    /**
     * The noisy networks parameters epsilon are re-sampled
     * before every action too, as explained in the section 3.1
     * of the paper.
     * @param state actual state
     * @return selected action
     */
    @Override
    public ActionSpaceType.ActionResult selectAction(NDArray state) {
        onlineRainbowNet.resetNoise();
        @Cleanup var oneBatchState = state.expandDims(0);
        long action = onlineRainbowNet.forwardLong(oneBatchState,
                qValue -> qValue.stopGradient().argMax(AXIS_1));
        return actionSpaceType.get(action);
    }

    /**
     * Updates the importance sampling annealing parameter (beta) based on the current training progress.
     *
     * <p>Beta is annealed from {@code initialBeta} to 1.0 over the course of training to
     * gradually correct the bias introduced by prioritized experience replay. This follows
     * the approach described in the Prioritized Experience Replay paper (Schaul et al., 2015).
     *
     * <p>The annealing formula is:
     * \[ \beta = \beta_{initial} + \text{fraction} \times (1.0 - \beta_{initial}) \]
     *
     * @throws IllegalArgumentException if {@code totalFramesLimit} <= 0 or {@code framesSkip} < 0
     * @see <a href="https://arxiv.org/abs/1511.05952">Prioritized Experience Replay</a>
     */
    @Override
    protected void templateExtraProcessing(int frames, long frameLimit) {
        float fraction = Math.min((float) frames / frameLimit, 1.0f);
        this.beta = this.initialBeta + fraction * (1.0f - this.initialBeta);
    }

    @Override
    public void close() {
        super.close();
        subManager.close();
    }
}
