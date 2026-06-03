package SequenceProcessing.Mistral;

/**
 * Small configurable Mistral-style decoder configuration.
 * The defaults are intentionally tiny so the project can run on a laptop.
 */
public class MistralConfig {
    public final int vocabSize;
    public final int maxSequenceLength;
    public final int hiddenSize;
    public final int intermediateSize;
    public final int numLayers;
    public final int numAttentionHeads;
    public final int numKeyValueHeads;
    public final double rmsNormEps;
    public final long seed;

    public MistralConfig() {
        this(256, 64, 64, 128, 2, 4, 2, 1e-6, 42L);
    }

    public MistralConfig(int vocabSize, int maxSequenceLength, int hiddenSize, int intermediateSize,
                         int numLayers, int numAttentionHeads, int numKeyValueHeads,
                         double rmsNormEps, long seed) {
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numAttentionHeads");
        }
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException("numAttentionHeads must be divisible by numKeyValueHeads for grouped-query attention");
        }
        this.vocabSize = vocabSize;
        this.maxSequenceLength = maxSequenceLength;
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
        this.numKeyValueHeads = numKeyValueHeads;
        this.rmsNormEps = rmsNormEps;
        this.seed = seed;
    }

    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }

    public int keyValueGroupSize() {
        return numAttentionHeads / numKeyValueHeads;
    }
}
