package br.com.guialves.rflr.algorithms.nstepdqn;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.NStepExperienceReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStep;
import static br.com.guialves.rflr.djlutils.DJLUtils.*;

/**
 * DQN with n-step returns.
 *
 * <p>Uses an {@link NStepExperienceReplayBuffer} that, on every {@code store()} call, builds
 * the n-step return and writes a synthetic multistep transition to the underlying ring buffer.
 * The training step is identical to vanilla DQN but the bootstrap target is computed over
 * the n-step reward.
 *
 * <p>Reference: Sutton (1988) "Learning to predict by the methods of temporal differences";
 * used as one of the seven improvements in Rainbow DQN (Hessel et al., 2017).
 */
@Slf4j
public class AgentNStepDQN extends AbstractAgent {

    public AgentNStepDQN(float epsilon,
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
        if (!(ireplayBuffer instanceof NStepExperienceReplayBuffer replayBuffer)) {
            throw new IllegalArgumentException("You must pass NStepExperienceReplayBuffer!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize);
        @Cleanup var targetQValue = targetNet.forward(samples.nextStates(), nextQValue -> {
            var rewards = samples.rewards();
            var dones = samples.dones();

            // max Q(s', a')
            var maxNextQValue = nextQValue.max(AXIS_1_ARR, KEEP_DIMS);
            float gammaNBootstrap = (float) Math.pow(gamma, replayBuffer.nStep());
            // gamma^n * max Q(s', a')
            var discountNextQValue = maxNextQValue.mul(gammaNBootstrap);
            // (1 - done)
            var mask = dones.neg().add(1);
            // r + gamma * max Q(s', a') * (1 - done)
            return rewards
                    .add(discountNextQValue.mul(mask))
                    .stopGradient();
        });

        float lossItem = backwardLoss(sub, lossFunc, targetQValue, () -> {
            var states = samples.states();
            var actions = samples.actions();
            // y_hat = q_online(s, a)
            return onlineNet.forward(states, qValue -> qValue.gather(actions, AXIS_1));
        });

        trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
    }
}
