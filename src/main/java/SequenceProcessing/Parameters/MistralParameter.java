package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.Loss.Loss;
import ComputationalGraph.NeuralNetworkParameter;
import ComputationalGraph.Optimizer.Optimizer;

import java.io.Serializable;

/**
 * Hyper-parameters of a {@link SequenceProcessing.Mistral.MistralModel}.
 * <p>
 * The values mirror the building blocks of the original Mistral architecture
 * (a stack of pre-normalised decoder blocks that use Root-Mean-Square
 * normalization, rotary positional embeddings, grouped-query attention and a
 * SwiGLU feed-forward network). They are kept small by default so the model can
 * be trained on a laptop, but every value is configurable.
 * <p>
 * This class extends {@link NeuralNetworkParameter} so a {@code MistralModel}
 * plugs into the same {@code ComputationalGraph} training machinery (optimizer,
 * loss function, weight initialisation) as the other models of this project.
 */
public class MistralParameter extends NeuralNetworkParameter implements Serializable {

    private final int hiddenSize;
    private final int numLayers;
    private final int numAttentionHeads;
    private final int numKeyValueHeads;
    private final int intermediateSize;
    private final int classLabelSize;
    private final double rmsNormEps;

    /**
     * Creates a fully specified set of Mistral hyper-parameters.
     *
     * @param seed              Random seed used both for shuffling and weight initialisation.
     * @param epoch             Number of training passes over the training set.
     * @param optimizer         Optimizer used to update the learnable weights (e.g. AdamW).
     * @param initialization    Strategy used to initialise the weight tensors.
     * @param loss              Loss function attached to the output of the graph (e.g. cross entropy).
     * @param hiddenSize        Dimension of the token/residual stream. Must be divisible by {@code numAttentionHeads}.
     * @param numLayers         Number of stacked decoder blocks.
     * @param numAttentionHeads Number of query heads in the attention layer.
     * @param numKeyValueHeads  Number of key/value heads. Must divide {@code numAttentionHeads} (grouped-query attention).
     * @param intermediateSize  Hidden dimension of the SwiGLU feed-forward network.
     * @param classLabelSize    Number of output classes (the size of the prediction vocabulary).
     * @param rmsNormEps        Small constant added inside RMSNorm for numerical stability.
     */
    public MistralParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization, Loss loss,
                            int hiddenSize, int numLayers, int numAttentionHeads, int numKeyValueHeads,
                            int intermediateSize, int classLabelSize, double rmsNormEps) {
        super(seed, epoch, optimizer, initialization, loss, 0.0, -1);
        if (hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numAttentionHeads");
        }
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException("numAttentionHeads must be divisible by numKeyValueHeads for grouped-query attention");
        }
        if ((hiddenSize / numAttentionHeads) % 2 != 0) {
            throw new IllegalArgumentException("headDim (hiddenSize / numAttentionHeads) must be even for rotary embeddings");
        }
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
        this.numKeyValueHeads = numKeyValueHeads;
        this.intermediateSize = intermediateSize;
        this.classLabelSize = classLabelSize;
        this.rmsNormEps = rmsNormEps;
    }

    /**
     * @return Dimension of the residual stream (token embedding size).
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * @return Number of stacked decoder blocks.
     */
    public int getNumLayers() {
        return numLayers;
    }

    /**
     * @return Number of query heads used by the attention layer.
     */
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }

    /**
     * @return Number of key/value heads (shared across query heads for grouped-query attention).
     */
    public int getNumKeyValueHeads() {
        return numKeyValueHeads;
    }

    /**
     * @return Hidden dimension of the SwiGLU feed-forward network.
     */
    public int getIntermediateSize() {
        return intermediateSize;
    }

    /**
     * @return Number of output classes produced by the language-modeling head.
     */
    public int getClassLabelSize() {
        return classLabelSize;
    }

    /**
     * @return Epsilon added inside RMSNorm for numerical stability.
     */
    public double getRmsNormEps() {
        return rmsNormEps;
    }

    /**
     * Size of a single attention head.
     *
     * @return {@code hiddenSize / numAttentionHeads}.
     */
    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }

    /**
     * Number of query heads that share a single key/value head.
     *
     * @return {@code numAttentionHeads / numKeyValueHeads}.
     */
    public int keyValueGroupSize() {
        return numAttentionHeads / numKeyValueHeads;
    }
}
