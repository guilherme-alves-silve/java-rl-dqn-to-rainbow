package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.NoisyQNetworkMLP;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStep;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1_ARR;

/**
 * DQN agent that replaces ε-greedy exploration with parametric noise added to the weights
 * (NoisyNet). The agent still uses a target network with the same hard-update cadence as DQN
 * but does not anneal any ε.
 *
 * <p>Before every optimization step (and every action selection) the agent calls
 * {@link NoisyQNetworkMLP#resetNoise()} on both online and target networks to sample a fresh
 * noise realization.
 *
 * <p>Reference: <a href="https://arxiv.org/abs/1706.10295">Noisy Networks for Exploration</a>.
 */
@Slf4j
public class AgentNoisyNetDQN extends AbstractAgent {

    private final NoisyQNetworkMLP onlineNoisyNet;
    private final NoisyQNetworkMLP targetNoisyNet;

    public AgentNoisyNetDQN(float epsilon,
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
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, parent,
                networkFactory, plotTrackers, debugMemoryLeak);
        if (!(onlineNet instanceof NoisyQNetworkMLP)) {
            throw new IllegalArgumentException("Invalid network type! Must be of type NoisyNetworkMLP!");
        }

        this.onlineNoisyNet = (NoisyQNetworkMLP) onlineNet;
        this.targetNoisyNet = (NoisyQNetworkMLP) targetNet;
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
    @Override
    protected float trainOnline(int batchSize, IReplayBuffer replayBuffer, Loss lossFunc, NDManager sub) {
        if (!replayBuffer.enough(batchSize)) return Float.NaN;
        onlineNoisyNet.resetNoise();

        @Cleanup var samples = replayBuffer.sample(batchSize);
        @Cleanup var targetQValue = targetNoisyNet.forward(samples.nextStates(), nextQValue -> {
            // max Q(s', a')
            var maxNextQValue = nextQValue.max(AXIS_1_ARR, true);
            // gamma * max Q(s', a')
            var discountNextQValue = maxNextQValue.mul(gamma);
            // (1 - done)
            var mask = samples.dones().neg().add(1);
            // r + gamma * max Q(s', a') * (1 - done)
            return samples.rewards()
                    .add(discountNextQValue.mul(mask))
                    .stopGradient();
        });

        float lossItem = backwardLoss(sub, lossFunc, targetQValue, () -> {
            var states = samples.states();
            var actions = samples.actions();
            // y_hat = q_online(s, a)
            return onlineNoisyNet.forward(states, qValue -> qValue.gather(actions, AXIS_1));
        });

        trainStep(onlineNoisyNet.getBlock(), optimizer);
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
        onlineNoisyNet.resetNoise();
        @Cleanup var oneBatchState = state.expandDims(0);
        long action = onlineNoisyNet.forwardLong(oneBatchState,
                qValue -> qValue.stopGradient().argMax(AXIS_1));
        return actionSpaceType.get(action);
    }
}
