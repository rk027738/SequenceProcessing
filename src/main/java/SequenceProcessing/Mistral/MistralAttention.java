package SequenceProcessing.Mistral;

import java.util.Random;

/** Masked multi-head self-attention with grouped-query attention. */
public class MistralAttention {
    private final MistralConfig config;
        /**
     * Query projection matrix.
     */
    private final double[][] queryWeights;
    
    /**
     * Key projection matrix.
     */
    private final double[][] keyWeights;
    
    /**
     * Value projection matrix.
     */
    private final double[][] valueWeights;
    
    /**
     * Output projection matrix.
     */
    private final double[][] outputWeights;
    private final RotaryEmbedding rope;

    public MistralAttention(MistralConfig config, Random random) {
        this.config = config;
        int kvSize = config.numKeyValueHeads * config.headDim();
        this.wq = TensorUtils.randomMatrix(config.hiddenSize, config.hiddenSize, random);
        this.wk = TensorUtils.randomMatrix(config.hiddenSize, kvSize, random);
        this.wv = TensorUtils.randomMatrix(config.hiddenSize, kvSize, random);
        this.wo = TensorUtils.randomMatrix(config.hiddenSize, config.hiddenSize, random);
        this.rope = new RotaryEmbedding(config.headDim());
    }

    public double[][] forward(double[][] x) {
        int seq = x.length;
        int headDim = config.headDim();
        double[][] qFlat = TensorUtils.matMul(x, wq);
        double[][] kFlat = TensorUtils.matMul(x, wk);
        double[][] vFlat = TensorUtils.matMul(x, wv);

        double[][] out = new double[seq][config.hiddenSize];
        for (int t = 0; t < seq; t++) {
            for (int h = 0; h < config.numAttentionHeads; h++) {
                int kvHead = h / config.keyValueGroupSize();
                double[] q = slice(qFlat[t], h * headDim, headDim);
                rope.applyInPlace(q, t);

                double[] scores = new double[t + 1];
                for (int s = 0; s <= t; s++) {
                    double[] k = slice(kFlat[s], kvHead * headDim, headDim);
                    rope.applyInPlace(k, s);
                    double dot = 0.0;
                    for (int d = 0; d < headDim; d++) dot += q[d] * k[d];
                    scores[s] = dot / Math.sqrt(headDim);
                }
                double[] probs = TensorUtils.softmax(scores);
                for (int s = 0; s <= t; s++) {
                    double[] v = slice(vFlat[s], kvHead * headDim, headDim);
                    for (int d = 0; d < headDim; d++) {
                        out[t][h * headDim + d] += probs[s] * v[d];
                    }
                }
            }
        }
        return TensorUtils.matMul(out, wo);
    }

    private static double[] slice(double[] values, int start, int length) {
        double[] out = new double[length];
        System.arraycopy(values, start, out, 0, length);
        return out;
    }
}
