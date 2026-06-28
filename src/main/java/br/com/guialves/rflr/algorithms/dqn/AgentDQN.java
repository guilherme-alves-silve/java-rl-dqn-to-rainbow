package br.com.guialves.rflr.algorithms.dqn;

import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.buffer.ExperienceReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;

@Slf4j
public class AgentDQN extends AbstractAgent {

    private final int[] the2ndAxis = new int[] {1};

    public AgentDQN(float epsilon, int updateQTargetAtTimeN,
                    float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Optimizer optimizer,
                    Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, networkFactory, plotTrackers);
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
    protected float trainOnline(int batchSize,
                                ExperienceReplayBuffer replayBuffer,
                                Loss lossFunc) {
        if (replayBuffer.size() < batchSize) return Float.NaN;

        @Cleanup var samples = replayBuffer.sample(batchSize);

        @Cleanup var targetQValue = scoped(arrays -> {
            var rewards = arrays[0];
            var nextStates = arrays[1];
            var dones = arrays[2];

            return targetNet.forward(nextStates, nextQValue -> {
                // max Q(s', a')
                var maxNextQValue = nextQValue.max(the2ndAxis, true);
                // gamma * max Q(s', a')
                var discountNextQValue = maxNextQValue.mul(gamma);
                // (1 - done)
                var mask = dones.neg().add(1);
                // r + gamma * max Q(s', a') * (1 - done)
                return rewards
                        .add(discountNextQValue.mul(mask))
                        .stopGradient();
            });
        }, samples.rewards(), samples.nextStates(), samples.dones());

        float lossItem = backwardLoss(env.manager(), lossFunc, targetQValue, arrays -> {
            var states = arrays[0];
            var actions = arrays[1];

            return onlineNet.forward(states, qValue ->
                    qValue.gather(actions, 1));
        }, samples.states(), samples.actions());

        DJLOptimizer.trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
    }
}
