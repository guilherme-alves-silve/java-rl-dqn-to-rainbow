package br.com.guialves.rflr.algorithms.noisydqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.algorithms.networks.NoisyDuelingQNetworkMLP;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;

@Slf4j
public class AgentNoisyDuelingNetDDQN extends AbstractAgent {

    private final int[] the2ndAxis = new int[] {1};
    private final NoisyDuelingQNetworkMLP onlineNoisyDuelNet;
    private final NoisyDuelingQNetworkMLP targetNoisyDuelNet;

    public AgentNoisyDuelingNetDDQN(float epsilon,
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
        if (!(onlineNet instanceof NoisyDuelingQNetworkMLP)) {
            throw new IllegalArgumentException("Invalid network type! Must be of type NoisyDuelingQNetworkMLP!");
        }

        this.onlineNoisyDuelNet = (NoisyDuelingQNetworkMLP) onlineNet;
        this.targetNoisyDuelNet = (NoisyDuelingQNetworkMLP) targetNet;
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

        @Cleanup var samples = replayBuffer.sample(batchSize);

        // reset first time eps' for DDQN
        onlineNoisyDuelNet.resetNoise();
        // a* = arg max q_online(s', a')
        var nextStates = samples.nextStates();
        @Cleanup var action = onlineNoisyDuelNet.forward(nextStates,
                qOnlineNext -> qOnlineNext.argMax(1).reshape(N_BATCH, 1));

        // reset first time eps' for DDQN
        targetNoisyDuelNet.resetNoise();
        @Cleanup var targetQValue = targetNoisyDuelNet.forward(nextStates, qNextValues -> {
            // q_target(s', a*)
            var qNextValue = qNextValues.gather(action, 1);
            // gamma * q_target(s', a*)
            var discountNextQValue = qNextValue.mul(gamma);
            // (1 - done)
            var mask = samples.dones().neg().add(1);
            // y = r + gamma * q_target(s', arg max q_online(s', a')) * (1 - done)
            return samples.rewards()
                    .add(discountNextQValue.mul(mask))
                    .stopGradient();
        });

        // reset second time eps' for DDQN now to compute TD-Error
        onlineNoisyDuelNet.resetNoise();
        float lossItem = backwardLoss(sub, lossFunc, targetQValue, () -> {
            var states = samples.states();
            var actions = samples.actions();
            // y_hat = q_online(s, a)
            return onlineNoisyDuelNet.forward(states, qValue -> qValue.gather(actions, 1));
        });

        DJLOptimizer.trainStep(onlineNoisyDuelNet.getBlock(), optimizer);
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
        onlineNoisyDuelNet.resetNoise();
        @Cleanup var oneBatchState = state.expandDims(0);
        @Cleanup var output = onlineNoisyDuelNet.forward(oneBatchState,
                qValue -> qValue.stopGradient().argMax(1));
        return actionSpaceType.get(output.getLong(0));
    }
}
