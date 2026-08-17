package br.com.guialves.rflr.algorithms.ddqn;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStep;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1;
import static br.com.guialves.rflr.djlutils.DJLUtils.N_BATCH;

@Slf4j
public class AgentDDQN extends AbstractAgent {

    public AgentDDQN(float epsilon,
                     int updateQTargetAtTimeN,
                     float minEpsilon,
                     float epsilonDecay,
                     float gamma, IEnv env,
                     Optimizer optimizer,
                     NDManager parent,
                     Supplier<IDeepQNetwork> networkFactory, PlotTrackers plotTrackers,
                     boolean debugMemoryLeak) {
        super(epsilon, updateQTargetAtTimeN, minEpsilon, epsilonDecay,
                gamma, env, optimizer, parent, networkFactory, plotTrackers, debugMemoryLeak);
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

        // a* = arg max q_online(s', a')
        var nextStates = samples.nextStates();
        @Cleanup var action = onlineNet.forward(nextStates,
                onlineQNextValue -> onlineQNextValue.stopGradient().argMax(AXIS_1)
                        .reshape(N_BATCH, 1));
        @Cleanup var targetQValue = targetNet.forward(nextStates, nextQValues -> {
            // q_target(s', a*) - DDQN
            var qNextValue = nextQValues.gather(action, AXIS_1);
            // gamma * q_target(s', a*)
            var discountNextQValue = qNextValue.mul(gamma);
            // (1 - done)
            var mask = samples.dones().neg().add(1);
            // y = r + gamma * q_target(s', arg max q_online(s', a')) * (1 - done)
            return samples.rewards()
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
