package SequenceProcessing.Mistral;

import java.util.Random;

/** One transformer decoder block: RMSNorm -> masked GQA -> residual -> RMSNorm -> SwiGLU -> residual. */
public class MistralDecoderBlock {
    private final RMSNorm attentionNorm;
    private final RMSNorm ffnNorm;
    private final MistralAttention attention;
    private final SwiGLUFeedForward feedForward;

    public MistralDecoderBlock(MistralConfig config, Random random) {
        this.attentionNorm = new RMSNorm(config.hiddenSize, config.rmsNormEps);
        this.ffnNorm = new RMSNorm(config.hiddenSize, config.rmsNormEps);
        this.attention = new MistralAttention(config, random);
        this.feedForward = new SwiGLUFeedForward(config, random);
    }

    public double[][] forward(double[][] x) {
        double[][] attentionOut = attention.forward(attentionNorm.forward(x));
        double[][] afterAttention = TensorUtils.add(x, attentionOut);
        double[][] ffnOut = feedForward.forward(ffnNorm.forward(afterAttention));
        return TensorUtils.add(afterAttention, ffnOut);
    }
}
