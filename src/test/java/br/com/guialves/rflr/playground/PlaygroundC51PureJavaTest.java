package br.com.guialves.rflr.playground;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;
import java.util.WeakHashMap;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class PlaygroundC51PureJavaTest {

    private static final double DELTA = 0.000001;

    /**
     * The idea of this playground is to be
     * easy to test and learn, so you can
     * understand step by step how the algorithm works.
     * r = reward
     * gamma = discount factor
     * z = support vector
     * deltaz = angular coefficient of z
     * Vmin = support vector minimum value
     * Vmax = support vector maximum value
     */
    @ParameterizedTest(name = "[{index}] {arguments}")
    @CsvSource(useHeadersInDisplayName = true, textBlock = """
    done
    0
    1
    """)
    void testDistributionalBellman(int done) {
        var expectedSupportVectorZ = new double[]{
                -10.0, -9.6, -9.2, -8.8, -8.4, -8.0, -7.6, -7.2, -6.8, -6.4,
                -6.0, -5.6, -5.2, -4.8, -4.4, -4.0, -3.6, -3.2, -2.8, -2.4,
                -2.0, -1.6, -1.2, -0.8, -0.4,  0.0,  0.4,  0.8,  1.2,  1.6,
                2.0,  2.4,  2.8,  3.2,  3.6,  4.0,  4.4,  4.8,  5.2,  5.6,
                6.0,  6.4,  6.8,  7.2,  7.6,  8.0,  8.4,  8.8,  9.2,  9.6,
                10.0
        };

        int state = 41;
        int action = 0;
        int nextState = 42;
        int nextAction = 1;

        double r = 0.5;
        double gamma = 0.9;
        double vmin = -10;
        double vmax = 10;
        int atoms = 51;

        var onlineNet = new NetworkMock(atoms, vmin, vmax);
        var targetNet = onlineNet.clone();

        double[] m = targetNet.bellmanProjection(done, nextState, nextAction, r, gamma);

        double[] predictedDist = onlineNet.predictDistribution(state, action);
        // L = -sum(mi * log(pi(s, a, theta)))
        double loss = crossEntropyLoss(m, predictedDist);
        IO.println("loss: " + loss);

        onlineNet.backward(loss);
        // Q-value = sum(zi * pi(s, a, theta))
        double qValue = onlineNet.predictQValue(state, action);
        IO.println("Q-Value: " + qValue);

        assertArrayEquals(expectedSupportVectorZ, onlineNet.supportVectorZ(), DELTA);
        assertEquals(1.0, Arrays.stream(predictedDist).sum(), DELTA);
        assertEquals(0.4, onlineNet.deltaz(), DELTA);
    }

    private static double[] supportVector(double vmin, double vmax, int atoms, double deltaz) {
        var z = new double[atoms];
        for (int i = 0; i < z.length; ++i) {
            z[i] = vmin + i * deltaz;
        }

        return z;
    }

    private static double crossEntropyLoss(double[] m, double[] predictedDist) {
        double sum = 0;
        for (int i = 0; i < m.length; ++i) {
            var stablePrediction = Math.max(predictedDist[i], DELTA);
            sum += m[i] * Math.log(stablePrediction);
        }
        return -sum;
    }

    private static double clip(double val, double lower, double upper) {
        if (val < lower) return lower;
        return Math.min(val, upper);
    }

    private static class NetworkMock implements Cloneable {

        private final int atoms;
        private final double vmin;
        private final double vmax;
        private final double deltaz;
        private final double[] z;
        private WeakHashMap<String, double[]> memoValues = new WeakHashMap<>();

        public NetworkMock(int atoms, double vmin, double vmax) {
            this.atoms = atoms;
            this.vmin = vmin;
            this.vmax = vmax;
            // delta_z = (Vmax - Vmin)/(N - 1)
            this.deltaz = (vmax - vmin)/(atoms - 1);
            // zi = { Vmin + i * delta_z : 0 <= i <= N - 1, i in N }
            this.z = supportVector(vmin, vmax, atoms, deltaz);
        }

        public double deltaz() {
            return deltaz;
        }

        public double[] supportVectorZ() {
            return z;
        }

        public double[] bellmanProjection(int done, int nextState, int nextAction, double r, double gamma) {

            double[] dist = this.predictDistribution(nextState, nextAction);
            double[] m = new double[atoms];
            for (int j = 0; j < atoms; ++j) {
                // Tzj = [r + gamma * zj * (1 - done)]_Vmin^Vmax
                double tzj = clip(r + gamma*z[j] * (1 - done), vmin, vmax);
                // b = (Tzj - vmin)/deltaz
                double b = (tzj - vmin)/deltaz;
                // l = floor(b)
                int l = (int) Math.floor(b);
                // u = ceil(b)
                int u = (int) Math.ceil(b);

                double prob = dist[j];
                if (l == u) {
                    m[l] = m[l] + prob;
                } else {
                    // ml = ml + p(s', a', theta^-) * (u - b)
                    m[l] = m[l] + prob * (u - b);
                    // mu = mu + p(s', a', theta^-) * (b - l)
                    m[u] = m[u] + prob * (b - l);
                }
            }

            // reward and gamma were projected into the distribution,
            // now online network can learn the Q-distribution
            return m;
        }

        public double[] predictDistribution(int state, int action) {
            // just to make the test easier to follow
            var key = state + "," + action;
            if (memoValues.containsKey(key)) return memoValues.get(key);
            var dist = gaussianGenerator();
            memoValues.put(key, dist);
            return dist;
        }

        private double[] gaussianGenerator() {
            var dist = new double[atoms];
            var mean = Math.random() * atoms;
            var stddev = DELTA + (Math.random() * 2);
            // gaussian distribution (mock only)
            // 1/(stddev * sqrt(2*number pi)) * exp(-(x - mu)^2/(2*stddev^2))
            var sum = 0.0;
            for (int i = 0; i < dist.length; ++i) {
                var numerator = -Math.pow((i - mean), 2);
                var denominator = 2*Math.pow(stddev, 2);
                dist[i] = 1/(stddev * Math.sqrt(2*Math.PI)) * Math.exp(numerator/denominator);
                sum += dist[i];
            }

            for (int i = 0; i < dist.length; ++i) {
                dist[i] /= sum;
            }

            // softmax/log-softmax simulation (sum must be 1)
            return dist;
        }

        /**
         * Used just to show that the target network is cloned
         * in the real implementation
         * @return new network based on the original net
         */
        @Override
        public NetworkMock clone() {
            try {
                var cloned = (NetworkMock) super.clone();
                cloned.memoValues = new WeakHashMap<>();
                return cloned;
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        public void backward(double loss) {
            IO.println("Doing backpropagation: " + loss);
        }

        public double predictQValue(int state, int action) {
            var dist = predictDistribution(state, action);
            double[] z = supportVectorZ();
            double sum = 0;
            for (int i = 0; i < dist.length; ++i) {
                sum += z[i] * dist[i];
            }

            return sum;
        }
    }
}
