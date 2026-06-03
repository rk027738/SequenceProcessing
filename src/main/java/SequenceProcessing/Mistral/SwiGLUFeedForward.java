package SequenceProcessing.Mistral;

import java.util.Random;

/** Mistral feed-forward network */
public class SwiGLUFeedForward {
    private final double[][] gate;
    private final double[][] up;
    private final double[][] down;

    public SwiGLUFeedForward(MistralConfig config, Random random) {
        this.gate = TensorUtils.randomMatrix(config.hiddenSize, config.intermediateSize, random);
        this.up = TensorUtils.randomMatrix(config.hiddenSize, config.intermediateSize, random);
        this.down = TensorUtils.randomMatrix(config.intermediateSize, config.hiddenSize, random);
    }

    public double[][] forward(double[][] x) {
        double[][] g = TensorUtils.matMul(x, gate);
        double[][] u = TensorUtils.matMul(x, up);
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[0].length; j++) g[i][j] = TensorUtils.silu(g[i][j]) * u[i][j];
        }
        return TensorUtils.matMul(g, down);
    }
}
