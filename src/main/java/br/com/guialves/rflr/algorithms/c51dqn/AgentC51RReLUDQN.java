package br.com.guialves.rflr.algorithms.c51dqn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import br.com.guialves.rflr.algorithms.AbstractAgent;
import br.com.guialves.rflr.algorithms.buffer.IReplayBuffer;
import br.com.guialves.rflr.algorithms.networks.CategoricalQNetworkRReLUMLP;
import br.com.guialves.rflr.algorithms.networks.IDeepQNetwork;
import br.com.guialves.rflr.gymnasium4j.ActionSpaceType;
import br.com.guialves.rflr.gymnasium4j.IEnv;
import br.com.guialves.rflr.utils.dataviz.PlotTrackers;
import lombok.Cleanup;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

import static br.com.guialves.rflr.djlutils.DJLLoss.backwardLoss;
import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;
import static br.com.guialves.rflr.djlutils.DJLOptimizer.trainStepClipGradients;
import static br.com.guialves.rflr.djlutils.DJLUtils.AXIS_1;
import static br.com.guialves.rflr.djlutils.DJLUtils.N_BATCH;

@Slf4j
public class AgentC51RReLUDQN extends AbstractAgent {

    private static final float CLIP_GRAD_THRESHOLD = 10.0f;

    private final CategoricalQNetworkRReLUMLP onlineCatNet;
    private final CategoricalQNetworkRReLUMLP targetCatNet;
    private final NDArray atomsBroadcaster;
    private final NDManager subManager;

    public AgentC51RReLUDQN(float epsilon,
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
        if (!(onlineNet instanceof CategoricalQNetworkRReLUMLP)) {
            throw new IllegalArgumentException("Invalid network type! Must be of type CategoricalQNetworkRReLUMLP!");
        }

        this.onlineCatNet = (CategoricalQNetworkRReLUMLP) onlineNet;
        this.targetCatNet = (CategoricalQNetworkRReLUMLP) targetNet;
        this.subManager = subMgr(parent, "sub-atoms-broadcast");
        this.atomsBroadcaster = this.targetCatNet.newAtomsBroadcaster(subManager);
    }

    /**
     * C51 Distributional DQN Algorithm Implementation
     *
     * <p><b>C51 Parameters:</b>
     * <br>V<sub>min</sub> = -10, V<sub>max</sub> = +10, atoms = 51
     * <br>Δz = (V<sub>max</sub> - V<sub>min</sub>) / (N - 1)
     *
     * <p><b>Support vector parameter:</b>
     * <br>z<sub>i</sub> = V<sub>min</sub> + i·Δz, &nbsp; {i ∈ ℤ | 0, 1, ..., N - 1}
     *
     * <p><b>C51 Bellman Projection:</b>
     * <br>T̂z<sub>j</sub> = [r + γ·z<sub>j</sub>]<sup>V<sub>max</sub></sup><sub>V<sub>min</sub></sub>
     * <br>b = (T̂z<sub>j</sub> - V<sub>min</sub>) / Δz
     * <br>l = ⌊b⌋
     * <br>u = ⌈b⌉
     * <br>m<sub>l</sub> ← m<sub>l</sub> + p<sub>j</sub>(s', a; θ<sup>target</sup>) · (u - b)
     * <br>m<sub>u</sub> ← m<sub>u</sub> + p<sub>j</sub>(s', a; θ<sup>target</sup>) · (b - l)
     * <br>if (l = u) then:
     * <br>m<sub>l</sub> ← m<sub>l</sub> + p<sub>j</sub>(s', a; θ<sup>target</sup>)
     *
     * <p><b>C51 Loss:</b>
     * <br><b>Training:</b>
     * <br>ℒ = -∑<sub>i=0</sub><sup>N-1</sup> m<sub>i</sub> · ln(p<sub>i</sub>(s, a; θ<sup>online</sup>))
     * <br><b>Inference:</b>
     * <br>ŷ = arg max<sub>a</sub> ∑<sub>i=0</sub><sup>N-1</sup> z<sub>i</sub> · p<sub>i</sub>(s, a; θ<sup>online</sup>)
     *
     * <p>Reference:
     * <a href="https://d2l.djl.ai/chapter_linear-networks/linear-regression-scratch.html">
     * DJL Linear Regression from Scratch</a>
     *
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

        if (!(lossFunc instanceof CategoricalNLLLoss)) {
            throw new IllegalArgumentException("You must pass CategoricalCrossEntropyLoss!");
        }

        @Cleanup var samples = replayBuffer.sample(batchSize);
        @Cleanup var projectDist = targetCatNet.forwardDist(samples.nextStates(), probNextDist -> {
            // (batch, actions, atoms) -> (batch, actions)
            var nextQValues = targetCatNet.qValuesFromDist(probNextDist);
            var bestNextActions = nextQValues.argMax(AXIS_1)
                    // (batch, 1, 1)
                    .reshape(N_BATCH, 1, 1)
                    // (batch, 1, atoms)
                    .mul(atomsBroadcaster);
            // (batch, 1, atoms) - now we are really selecting only actions a*, not all actions -> p(s', a*, theta-).
            var bestNextProbDist = probNextDist.gather(bestNextActions, AXIS_1).stopGradient();
            // Bellman Projection - mi
            return targetCatNet.projectBellman(bestNextProbDist, samples.rewards(), samples.dones(), gamma);
        });

        // Loss = -sum mi * ln (p(s, a, theta))
        float lossItem = backwardLoss(sub, lossFunc, projectDist, array -> {
            var states = array[0];
            var actions = array[1]
                    // (batch, 1, 1)
                    .reshape(N_BATCH, 1, 1)
                    // (batch, 1, atoms)
                    .mul(atomsBroadcaster);
            // ln(p(s, a, theta))
            return onlineCatNet.forwardLogits(states, logits -> logits.gather(actions, AXIS_1));
        }, samples.states(), samples.actions());

        trainStepClipGradients(onlineNet.getBlock(), optimizer, CLIP_GRAD_THRESHOLD);
        return lossItem;
    }

    @Override
    public ActionSpaceType.ActionResult selectAction(NDArray state) {
        if (!test) {
            var rand = ThreadLocalRandom.current().nextFloat();
            if (rand < epsilon) {
                return env.actionSpaceSample();
            }
        }

        @Cleanup var oneBatchState = state.expandDims(0);
        try {
            // Because o RReLU
            onlineNet.eval();
            long action = onlineNet.forwardLong(oneBatchState,
                    qValue -> qValue.stopGradient().argMax(AXIS_1));
            return actionSpaceType.get(action);
        } finally {
            onlineNet.train();
        }
    }

    @Override
    public void close() {
        super.close();
        subManager.close();
    }
}
