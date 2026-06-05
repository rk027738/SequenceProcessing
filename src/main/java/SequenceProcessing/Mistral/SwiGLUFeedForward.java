package SequenceProcessing.Mistral;

import java.util.Random;

/** Mistral feed-forward network */
public class SwiGLUFeedForward {
    /** Gate projection used by SwiGLU. */
    private final double[][] gateProjection;
    /** Expansion projection. */
    private final double[][] expansionProjection;
    /** Compression projection back to hidden size. */
    private final double[][] compressionProjection;

    public SwiGLUFeedForward(MistralConfig config, Random random) {
        this.gateProjection = TensorUtils.randomMatrix(config.hiddenSize, config.intermediateSize, random);
        this.expansionProjection = TensorUtils.randomMatrix(config.hiddenSize, config.intermediateSize, random);
        this.compressionProjection = TensorUtils.randomMatrix(config.intermediateSize, config.hiddenSize, random);
    }

    public double[][] forward(double[][] x) {
        double[][] g = TensorUtils.matMul(x, gateProjection);
        double[][] u = TensorUtils.matMul(x, expansionProjection);
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[0].length; j++) g[i][j] = TensorUtils.silu(g[i][j]) * u[i][j];
        }
        return TensorUtils.matMul(g, compressionProjection);
    }
}
