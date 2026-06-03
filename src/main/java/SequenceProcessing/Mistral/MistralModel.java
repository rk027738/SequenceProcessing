package SequenceProcessing.Mistral;

import java.util.Random;

/**
 * Mistral-style decoder-only language model implementation
 */
public class MistralModel {
    private final MistralConfig config;
    private final double[][] tokenEmbedding;
    private final MistralDecoderBlock[] layers;
    private final RMSNorm finalNorm;
    private final double[][] lmHead;

    public MistralModel(MistralConfig config) {
        this.config = config;
        Random random = new Random(config.seed);
        this.tokenEmbedding = TensorUtils.randomMatrix(config.vocabSize, config.hiddenSize, random);
        this.layers = new MistralDecoderBlock[config.numLayers];
        for (int i = 0; i < layers.length; i++) layers[i] = new MistralDecoderBlock(config, random);
        this.finalNorm = new RMSNorm(config.hiddenSize, config.rmsNormEps);
        this.lmHead = TensorUtils.randomMatrix(config.hiddenSize, config.vocabSize, random);
    }

    public double[][] forward(int[] tokens) {
        if (tokens.length == 0) throw new IllegalArgumentException("At least one token is required");
        if (tokens.length > config.maxSequenceLength) throw new IllegalArgumentException("Input is longer than maxSequenceLength");
        double[][] x = embed(tokens);
        for (MistralDecoderBlock layer : layers) x = layer.forward(x);
        x = finalNorm.forward(x);
        double[][] logits = TensorUtils.matMul(x, lmHead);
        double[][] probs = new double[logits.length][config.vocabSize];
        for (int i = 0; i < logits.length; i++) probs[i] = TensorUtils.softmax(logits[i]);
        return probs;
    }

    public int predictNextToken(int[] tokens) {
        double[][] probs = forward(tokens);
        double[] last = probs[probs.length - 1];
        int best = 0;
        for (int i = 1; i < last.length; i++) if (last[i] > last[best]) best = i;
        return best;
    }

    private double[][] embed(int[] tokens) {
        double[][] x = new double[tokens.length][config.hiddenSize];
        for (int i = 0; i < tokens.length; i++) {
            int token = Math.floorMod(tokens[i], config.vocabSize);
            System.arraycopy(tokenEmbedding[token], 0, x[i], 0, config.hiddenSize);
        }
        return x;
    }
}
