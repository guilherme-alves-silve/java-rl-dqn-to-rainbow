package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.djlutils.DJLOptimizer;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;

@Slf4j
public class AgentC51DQN extends AbstractAgent {

    private final int[] the2ndAxis = new int[] {1};

    public AgentC51DQN(float epsilon,
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
     * \begin{align}
     * & \text{C51 Parameters:}\\
     * & \quad V_{\min} = -10, V_{\max} = +10, \text{atoms} = 51 \\
     * & \quad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1} \\
     * & \text{Support vector parameter:}\\
     * & \quad z_i = V_{\min} + i \Delta z \, , \quad \{i \in \mathbb{Z} \, | \, 0, 1, \cdots, N - 1  \} \\ \\
     * & \text{C51 Bellman Projection:} \\
     * & \hat{T}z_j = [r + \gamma z_j]^{V_{\max}}_{V_{\min}} \\
     * & b = \frac{\hat{T}z_j  - V_{\min}}{\Delta z} \\
     * & l = \lfloor b \rfloor \\
     * & u = \lceil b \rceil \\
     * & m_l \leftarrow m_l + p_j(s', a; \theta^{target}) \cdot (u - b) \\
     * & m_u \leftarrow m_u + p_j(s', a; \theta^{target}) \cdot (b - l) \\
     * & \text{if } (l = u) \text{ then: } \\
     * & \quad m_l \leftarrow m_l + p_j(s', a; \theta^{target}) \\ \\
     * & \text{C51 Loss with PER:} \\
     * & \text{Training:} \\
     * & \mathcal{L} = -\sum\limits_k \overline{w}^{\tiny IS}_k \sum\limits_{i=0}^{N-1} m_i \cdot \ln (p_i(s, a; \theta^{online})) \\
     * & \text{Inference:} \\
     * & \hat{y} = \arg \max\limits_a \sum\limits_{i=0}^{N - 1} z_i \cdot p_i(s, a; \theta^{online})
     * \end{align}
     * Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-scratch.html">DJL Linear Regression from Scratch</a>
     * @param batchSize Number of experiences to sample from the replay buffer for each training step
     * @param replayBuffer Experience replay buffer containing stored transitions (state, action, reward, nextState, done)
     * @param lossFunc Loss function used to compute the difference between current Q-values and target Q-values (e.g., MSE, Huber)
     */
    @Override
    protected float trainOnline(int batchSize,
                                IReplayBuffer replayBuffer,
                                Loss lossFunc,
                                NDManager sub) {
        if (!replayBuffer.enough(batchSize)) return Float.NaN;

        @Cleanup var samples = replayBuffer.sample(batchSize);
        @Cleanup var targetQValue = targetNet.forward(samples.nextStates(), nextQValue -> {
            // max Q(s', a')
            var maxNextQValue = nextQValue.max(the2ndAxis, true);
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
            return onlineNet.forward(states, qValue -> qValue.gather(actions, 1));
        });

        DJLOptimizer.trainStep(onlineNet.getBlock(), optimizer);
        return lossItem;
    }
}
