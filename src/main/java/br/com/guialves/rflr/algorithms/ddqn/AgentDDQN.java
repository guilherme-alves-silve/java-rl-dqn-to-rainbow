package br.com.guialves.rflr.algorithms.ddqn;

import ai.djl.ndarray.NDArray;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.Experience;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.EnvStepResult;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.scoped;

@Slf4j
public class AgentDDQN extends AbstractAgent<Experience> {

    public AgentDDQN(float epsilon, int updateQTargetAtTimeN,
                    float minEpsilon, float epsilonDecay,
                    float gamma, IEnv env, Optimizer optimizer,
                    Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers) {
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, networkFactory, plotTrackers);
    }

    @Override
    protected Experience newExperience(NDArray state,
                                       ActionSpaceType.ActionResult action,
                                       double reward,
                                       NDArray nextState,
                                       boolean done) {
        return new Experience(state, action, reward, nextState, done);
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
    protected float trainOnline(int batchSize, IReplayBuffer<Experience> replayBuffer, Loss lossFunc) {
        if (replayBuffer.size() < batchSize) return Float.NaN;

        @Cleanup var samples = replayBuffer.sample(batchSize);

        @Cleanup var targetQValue = scoped(arrays -> {
            var rewards = arrays[0];
            var nextStates = arrays[1];
            var dones = arrays[2];

            // a* = arg max q_online(s', a')
            @Cleanup var action = onlineNet.forward(nextStates,
                    qOnlineNext -> qOnlineNext.argMax(1).reshape(N_BATCH, 1));

            return targetNet.forward(nextStates, qNextValues -> {
                // q_target(s', a*)
                var qNextValue = qNextValues.gather(action, 1);
                // gamma * q_target(s', a*)
                var discountNextQValue = qNextValue.mul(gamma);
                // (1 - done)
                @Cleanup var mask = dones.neg().add(1);
                // y = r + gamma * q_target(s', arg max q_online(s', a')) * (1 - done)
                return rewards.add(discountNextQValue.mul(mask))
                        .stopGradient();
            });
        }, samples.rewards(), samples.nextStates(), samples.dones());

        float lossItem = backwardLoss(env.manager(), lossFunc, targetQValue, arrays -> {
            var states = arrays[0];
            var actions = arrays[1];
            // y_hat = q_online(s, a)
            return onlineNet.forward(states, qValue -> qValue.gather(actions, 1));
        }, samples.states(), samples.actions());

        DJLOptimizer.trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
    }
}
