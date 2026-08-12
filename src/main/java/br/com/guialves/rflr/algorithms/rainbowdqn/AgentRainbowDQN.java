package br.com.guialves.rflr.algorithms.rainbowdqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.NStepPrioritizedReplayBuffer;
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
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scopedToFloat;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStepClipGradients;
import static br.com.guialves.rflr.djlutils.DJLUtils.*;

@Slf4j
public class AgentRainbowDQN extends AbstractAgent {

    private static final float CLIP_GRAD_THRESHOLD = 10.0f;

    private final RainbowQNetworkMLP onlineRainbowNet;
    private final RainbowQNetworkMLP targetRainbowNet;
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
        if (!(ireplayBuffer instanceof NStepPrioritizedReplayBuffer replayBuffer)) {
            throw new IllegalArgumentException("You must pass PrioritizedReplayBuffer!");
        }
        if (!(lossFunc instanceof CategoricalCrossEntropyPERLoss catLossFunc)) {
            throw new IllegalArgumentException("You must pass CategoricalCrossEntropyPERLoss!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize);

        // reset first time eps' for DDQN
        onlineRainbowNet.resetNoise();
        // a* = arg max q_online(s', a')
        var nextStates = samples.nextStates();
        @Cleanup var action = onlineRainbowNet.forward(nextStates,
                qOnlineNext -> qOnlineNext.argMax(1).reshape(N_BATCH, 1));

        // reset first time eps' for DDQN
        targetRainbowNet.resetNoise();
        @Cleanup var targetQValue = targetRainbowNet.forward(nextStates, nextQValues -> {
            var rewards = samples.rewards();
            var dones = samples.dones();
            // q_target(s', a*)
            var nextQValue = nextQValues.gather(action, AXIS_1);
            float gammaNBootstramp = (float) Math.pow(gamma, replayBuffer.nStep());
            // gamma^n * max Q(s', a')
            var discountNextQValues = nextQValue.mul(gammaNBootstramp);
            // (1 - done)
            var mask = dones.neg().add(1);
            // r + gamma * max Q(s', a') * (1 - done)
            return rewards
                    .add(discountNextQValues.mul(mask))
                    .stopGradient();
        });

        // reset second time eps' for DDQN now to compute TD-Error
        onlineRainbowNet.resetNoise();
        catLossFunc.normISWeights(samples.weights());
        var losses = rawBackwardLoss(sub, catLossFunc, targetQValue, () -> {
            var states = samples.states();
            var actions = samples.actions();
            // y_hat = q_online(s, a)
            return onlineRainbowNet.forward(states, qValue -> qValue.gather(actions, AXIS_1));
        });

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
                qValue -> qValue.stopGradient().argMax(1));
        return actionSpaceType.get(action);
    }
}
