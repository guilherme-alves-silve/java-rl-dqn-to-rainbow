package br.com.guialves.rflr.algorithms.dqnper;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.rawBackwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scopedToFloat;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1_ARR;
import static br.com.guialves.rflr.djlutils.DJLUtils.KEEP_DIMS;

@Slf4j
public class AgentDQNPER extends AbstractAgent {

    private static final Float MIN_PRIORITY = 0.000_001f;

    private final float initialBeta;
    private float beta;

    public AgentDQNPER(float epsilon,
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
        this.initialBeta = beta;
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
        if (!(ireplayBuffer instanceof PrioritizedReplayBuffer replayBuffer)) {
            throw new IllegalArgumentException("You must pass PrioritizedReplayBuffer!");
        }
        if (!(lossFunc instanceof PERL2Loss perl2LossFunc)) {
            throw new IllegalArgumentException("You must pass PERL2Loss!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize, beta);
        @Cleanup var targetQValue = targetNet.forward(samples.nextStates(), nextQValue -> {
            // max Q(s', a')
            var maxNextQValue = nextQValue.max(AXIS_1_ARR, KEEP_DIMS);
            // gamma * max Q(s', a')
            var discountNextQValue = maxNextQValue.mul(gamma);
            // (1 - done)
            var mask = samples.dones().neg().add(1);
            // r + gamma * max Q(s', a') * (1 - done)
            return samples.rewards()
                    .add(discountNextQValue.mul(mask))
                    .stopGradient();
        });

        perl2LossFunc.normISWeights(samples.weights());
        // (perl2LossFunc) L = sum norm w^{is} * error^2
        @Cleanup var losses = rawBackwardLoss(sub, perl2LossFunc, targetQValue, () -> {
            var states = samples.states();
            var actions = samples.actions();
            // y_hat = q_online(s, a)
            return onlineNet.forward(states, qValue -> qValue.gather(actions, 1));
        });

        var lossItem = scopedToFloat(NDArray::mean, losses);
        @Cleanup var priorities = scoped(it -> it.abs().add(MIN_PRIORITY), losses);

        replayBuffer.updatePriorities(samples.bufferIndexes(), priorities);
        DJLOptimizer.trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
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
}
