package SequenceProcessing.Mistral;

import java.util.Random;

final class TensorUtils {
    private TensorUtils() {}

    static double[][] randomMatrix(int rows, int cols, Random random) {
        double[][] out = new double[rows][cols];
        double scale = Math.sqrt(2.0 / Math.max(1, rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = random.nextGaussian() * scale;
            }
        }
        return out;
    }

    static double[] ones(int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = 1.0;
        return out;
    }

    static double[] matVec(double[] x, double[][] w) {
        double[] y = new double[w[0].length];
        for (int j = 0; j < y.length; j++) {
            double sum = 0.0;
            for (int i = 0; i < x.length; i++) sum += x[i] * w[i][j];
            y[j] = sum;
        }
        return y;
    }

    static double[][] matMul(double[][] a, double[][] b) {
        double[][] out = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < b.length; k++) {
                double aik = a[i][k];
                for (int j = 0; j < b[0].length; j++) out[i][j] += aik * b[k][j];
            }
        }
        return out;
    }

    static double[][] add(double[][] a, double[][] b) {
        double[][] out = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) out[i][j] = a[i][j] + b[i][j];
        }
        return out;
    }

    static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;
        double sum = 0.0;
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }

    static double silu(double x) {
        return x / (1.0 + Math.exp(-x));
    }
}
