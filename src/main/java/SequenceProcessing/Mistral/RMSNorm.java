package SequenceProcessing.Mistral;

/** Root Mean Square normalization used by Mistral before attention and feed-forward blocks. */
public class RMSNorm {
    private final double[] weight;
    private final double eps;

    public RMSNorm(int hiddenSize, double eps) {
        this.weight = TensorUtils.ones(hiddenSize);
        this.eps = eps;
    }

    public double[] forward(double[] x) {
        double meanSquare = 0.0;
        for (double v : x) meanSquare += v * v;
        meanSquare /= x.length;
        double scale = 1.0 / Math.sqrt(meanSquare + eps);
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] * scale * weight[i];
        return out;
    }

    public double[][] forward(double[][] x) {
        double[][] out = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) out[i] = forward(x[i]);
        return out;
    }
}
