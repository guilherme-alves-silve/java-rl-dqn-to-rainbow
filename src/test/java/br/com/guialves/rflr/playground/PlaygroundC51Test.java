package br.com.guialves.rflr.playground;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;

public class PlaygroundC51Test {

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
    void testDistributionalBellmanTargetPureJava(int done) {
        double r = 0.5;
        double gamma = 0.9;
        double Vmin = -2;
        double Vmax = 2;
        int N = 5;
        double deltaz = (Vmax - Vmin)/(N - 1);
        double[] z = new double[N];
        for (int i = 0; i < N; ++i) {
            z[i] = Vmin + deltaz*i;
        }

        double[] Tz = new double[N];
        double[] b = new double[N];
        double[] l = new double[N];
        double[] u = new double[N];

        for (int j = 0; j < N; ++j) {
            Tz[j] = clip(r + gamma * z[j] * (1 - done), Vmin, Vmax);
            b[j] = (Tz[j] - Vmin)/deltaz;
            l[j] = Math.floor(b[j]);
            u[j] = Math.ceil(b[j]);
        }

        IO.println("z: " + Arrays.toString(z));
        IO.println("Tz: " + Arrays.toString(Tz));
        IO.println("b: " + Arrays.toString(b));
        IO.println("l: " + Arrays.toString(l));
        IO.println("u: " + Arrays.toString(u));
    }

    private static double clip(double val, double lower, double upper) {
        if (val < lower) return lower;
        return Math.min(val, upper);
    }
}
