package SequenceProcessing.Mistral;

/** Rotary positional embedding, applied to query and key vectors. */
public class RotaryEmbedding {
    private final int headDim;

    public RotaryEmbedding(int headDim) {
        if (headDim % 2 != 0) throw new IllegalArgumentException("headDim must be even for RoPE");
        this.headDim = headDim;
    }

    public void applyInPlace(double[] vector, int position) {
        for (int i = 0; i < headDim; i += 2) {
            double theta = Math.pow(10000.0, -((double) i / headDim));
            double angle = position * theta;
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            double x1 = vector[i];
            double x2 = vector[i + 1];
            vector[i] = x1 * cos - x2 * sin;
            vector[i + 1] = x1 * sin + x2 * cos;
        }
    }
}
