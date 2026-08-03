package br.com.guialves.rflr.algorithms.networks.distributional;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import lombok.Cleanup;
import lombok.Getter;
import lombok.experimental.Accessors;

import static br.com.guialves.rflr.djlutils.DJLMemoryManagement.subMgr;

@Accessors(fluent = true)
public class CategoricalBellmanProjection {

    public static final int N_ATOMS = 51;
    public static final int SELECTED_ACTION_PER_BATCH = 1;
    public static final float V_MIN = -10.0f;
    public static final float V_MAX = +10.0f;

    @Getter
    private final int atoms;
    @Getter
    private final float vMin;
    @Getter
    private final float vMax;
    @Getter
    private final float zDelta;
    private final float[] supportVectorZ;

    public CategoricalBellmanProjection() {
        this(N_ATOMS, V_MIN, V_MAX);
    }

    public CategoricalBellmanProjection(int atoms,
                                        float vMin,
                                        float vMax) {
        this.atoms = atoms;
        this.vMin = vMin;
        this.vMax = vMax;
        this.zDelta = (vMax - vMin)/(atoms - 1);
        this.supportVectorZ = generateSupportVectorZ(atoms, vMin, zDelta);
    }

    public float[] support() {
        return supportVectorZ.clone();
    }

    public NDArray support(NDManager mgr) {
        return mgr.create(supportVectorZ);
    }

    private float[] generateSupportVectorZ(int atoms, float vMin, float zDelta) {
        var zSupport = new float[atoms];
        for (int i = 0; i < atoms; ++i) {
            zSupport[i] = vMin + i * zDelta;
        }

        return zSupport;
    }

    /**
     * The shape (batchSize, 1, atoms) is because we select 1 actions between
     * the actions to project the bellman equation, so the training can happen
     * @param probNextDist The next distribution, got from target network
     * @param rewards The rewards of the actions given state s'
     * @param dones If it's done
     * @param gamma The discount factor
     * @return The projection onto the NDArray
     */
    public NDArray project(final NDArray probNextDist,
                           final NDArray rewards,
                           final NDArray dones,
                           final float gamma) {

        @Cleanup var sub = subMgr(probNextDist, "bellman-proj");
        sub.tempAttachAll(probNextDist, rewards, dones);

        int batchSize = (int) rewards.getShape().get(0);
        float[] rewardsArr = rewards.reshape(-1).toFloatArray();
        float[] probNextDistArr = probNextDist.reshape(-1).toFloatArray();
        float[] donesArr = dones.toFloatArray();
        float[] massDist = new float[batchSize * atoms];

        for (int batch = 0; batch < batchSize; ++batch) {
            int offset = batch * atoms;
            float reward = rewardsArr[batch];
            float notDone = 1 - donesArr[batch];
            for (int atomIdx = 0; atomIdx < atoms; ++atomIdx) {
                // Tz = r + γ * z_j * (1 - done)
                float targetProj = Math.clamp(reward + gamma * supportVectorZ[atomIdx] * notDone, vMin, vMax);
                // b = (Tz - vMin) / ∆z
                float baseIdx = (targetProj - vMin) / zDelta;
                int lowerIdx = (int) Math.floor(baseIdx);
                int upperIdx = (int) Math.ceil(baseIdx);
                // m_l += p(s', a, θ-) * (u - b)
                // m_u += p(s', a, θ-) * (b - l)
                if (lowerIdx == upperIdx) {
                    massDist[offset + lowerIdx] += probNextDistArr[atomIdx];
                } else {
                    massDist[offset + lowerIdx] += probNextDistArr[atomIdx] * (upperIdx - baseIdx);
                    massDist[offset + upperIdx] += probNextDistArr[atomIdx] * (baseIdx - lowerIdx);
                }
            }
        }

        var massDistTarget = sub.create(massDist, new Shape(batchSize, SELECTED_ACTION_PER_BATCH, atoms));
        return sub.ret(massDistTarget);
    }
}
