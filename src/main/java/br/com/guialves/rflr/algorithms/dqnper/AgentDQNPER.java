package br.com.guialves.rflr.algorithms.dqnper;

import ai.djl.ndarray.NDArray;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedExperience;
import br.com.guialves.rflr.algorithms.buffer.PrioritizedReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.rawBackwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scopedToFloat;

@Slf4j
public class AgentDQNPER extends AbstractAgent<PrioritizedExperience> {

    private static final Float MIN_PRIORITY = 0.000_001f;
    private final int[] the2ndAxis = new int[] {1};
    @Getter @Setter
    private float beta;

    public AgentDQNPER(float epsilon, int updateQTargetAtTimeN,
                       float minEpsilon, float epsilonDecay,
                       float gamma, IEnv env, Optimizer optimizer,
                       Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, networkFactory, plotTrackers);
    }

    @Override
    protected PrioritizedExperience newExperience(NDArray state,
                                                  ActionSpaceType.ActionResult action,
                                                  double reward,
                                                  NDArray nextState,
                                                  boolean done) {
        return new PrioritizedExperience(state, action, reward, nextState, done, MIN_PRIORITY);
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
    protected float trainOnline(int batchSize,
                                IReplayBuffer<PrioritizedExperience> ireplayBuffer,
                                Loss lossFunc) {
        if (ireplayBuffer.size() < batchSize) return Float.NaN;
        if (!(ireplayBuffer instanceof PrioritizedReplayBuffer replayBuffer)) {
            throw new IllegalArgumentException("You must pass PrioritizedReplayBuffer!");
        }
        if (!(lossFunc instanceof PERL2Loss perl2LossFunc)) {
            throw new IllegalArgumentException("You must pass PERL2Loss!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize, beta);
        @Cleanup var targetQValue = targetNet.forward(samples.nextStates(), (nextQValue, arrays) -> {
            var rewards = arrays[0];
            var dones = arrays[1];
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
        }, samples.rewards(), samples.dones());

        perl2LossFunc.normISWeights(samples.weights());
        // (perl2LossFunc) L = sum norm w^{is} * error^2
        @Cleanup var losses = rawBackwardLoss(env.manager(), perl2LossFunc, targetQValue, arrays -> {
            var states = arrays[0];
            var actions = arrays[1];
            // y_hat = q_online(s, a)
            return onlineNet.forward(states, qValue -> qValue.gather(actions, 1));
        }, samples.states(), samples.actions(), perl2LossFunc.normISWeights());

        var lossItem = scopedToFloat(it -> it.stopGradient().mean(), losses);
        var priorities = scoped(it -> it.abs().add(MIN_PRIORITY), losses);

        replayBuffer.updatePriorities(samples.bufferIndexes(), priorities);
        DJLOptimizer.trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
    }
}
