package SequenceProcessing.Mistral;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;
import ComputationalGraph.Function.SiLU;
import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.ConcatenatedNode;
import ComputationalGraph.Node.MultiplicationNode;
import Math.Tensor;
import SequenceProcessing.Functions.Inverse;
import SequenceProcessing.Functions.Mask;
import SequenceProcessing.Functions.MultiplyByConstant;
import SequenceProcessing.Functions.RotaryPositionalEmbedding;
import SequenceProcessing.Functions.SquareRoot;
import SequenceProcessing.Functions.Transpose;
import SequenceProcessing.Functions.Variance;
import SequenceProcessing.Parameters.MistralParameter;

import java.util.ArrayList;
import java.util.Random;

/**
 * A small Mistral-style language model.
 *
 * <p>The whole model is built with the ComputationalGraph library instead of doing
 * the maths by hand. The idea is simple: we describe the network as a graph of
 * small steps (multiply by a weight, normalise, apply softmax, ...). Once the
 * graph is built, the library does the forward pass, works out all the gradients
 * for us (back-propagation) and lets the optimizer update the weights. This is
 * the same approach used by the Transformer and the RNN models in this project.</p>
 *
 * <p>The model is a stack of decoder blocks. Each block does:</p>
 * <pre>
 *   RMSNorm -&gt; self attention -&gt; add the input back (residual)
 *   RMSNorm -&gt; feed forward    -&gt; add the input back (residual)
 * </pre>
 * <p>After the last block we do one more RMSNorm, multiply by an output weight to
 * get one score per class, and finally apply softmax to turn the scores into
 * probabilities. The pieces that make this "Mistral" rather than a plain
 * transformer are: RMSNorm (instead of layer norm), rotary position embeddings
 * (RoPE), grouped-query attention and a SwiGLU feed-forward network.</p>
 *
 * <p>The model works on a whole sentence at once: the input of the graph is a
 * matrix of size [number of tokens, hidden size]. Each training example is a flat
 * list of numbers where, for every token, we store its embedding followed by its
 * correct class label.</p>
 */
public class MistralModel extends ComputationalGraph {

    /** The settings of the model (sizes, number of layers, optimizer, ...). */
    private final MistralParameter parameter;

    /**
     * Creates the model. The graph itself is only built later, at the start of
     * {@link #train(ArrayList)}.
     *
     * @param parameter the model settings.
     */
    public MistralModel(MistralParameter parameter) {
        super(parameter);
        this.parameter = parameter;
    }

    /**
     * Adds one simple linear layer to the graph, i.e. it multiplies the input by a
     * learnable weight matrix (input @ weight). Mistral does not use bias terms,
     * so we don't add one.
     *
     * @param input      the node we want to project.
     * @param inputSize  number of input features.
     * @param outputSize number of output features.
     * @param random     random generator used to fill the weight matrix.
     * @return the node that holds input @ weight.
     */
    private ComputationalNode linearLayer(ComputationalNode input, int inputSize, int outputSize, Random random) {
        Tensor weight = new Tensor(parameter.initializeWeights(inputSize, outputSize, random), new int[]{inputSize, outputSize});
        return this.addEdge(input, new MultiplicationNode(weight));
    }

    /**
     * Adds RMSNorm on top of a node.
     *
     * <p>RMSNorm divides every row by the square root of the average of its squared
     * values, and then multiplies it by a learnable scale (gamma). In short:
     * {@code out = x / sqrt(mean(x^2) + eps) * gamma}. Unlike layer norm, it does
     * not subtract the mean first.</p>
     *
     * @param input the node to normalise.
     * @return the normalised node.
     */
    private ComputationalNode rmsNorm(ComputationalNode input) {
        int hiddenSize = parameter.getHiddenSize();
        ComputationalNode meanOfSquares = this.addEdge(input, new Variance());
        ComputationalNode root = this.addEdge(meanOfSquares, new SquareRoot(parameter.getRmsNormEps()));
        ComputationalNode oneOverRoot = this.addEdge(root, new Inverse());
        ComputationalNode normalised = this.addEdge(input, oneOverRoot, false, true);
        ArrayList<Double> gammaValues = new ArrayList<>();
        for (int i = 0; i < hiddenSize; i++) {
            gammaValues.add(1.0);
        }
        ComputationalNode gamma = new MultiplicationNode(true, false, new Tensor(gammaValues, new int[]{1, hiddenSize}), true);
        return this.addEdge(normalised, gamma);
    }

    /**
     * Adds the self-attention part of a block.
     *
     * <p>Each token looks at the previous tokens (and itself) to decide what is
     * important. We compute queries (Q), keys (K) and values (V) with linear
     * layers, add rotary position information to Q and K, and then score every
     * pair of tokens with Q times K. We hide the future with a causal mask, turn
     * the scores into weights with softmax, and use those weights to mix the
     * values.</p>
     *
     * <p>This uses grouped-query attention: there are fewer key/value heads than
     * query heads, so a group of query heads shares the same key/value head.</p>
     *
     * @param input  the (already normalised) input node.
     * @param random random generator used for the weights.
     * @return the attention output node.
     */
    private ComputationalNode selfAttention(ComputationalNode input, Random random) {
        int hiddenSize = parameter.getHiddenSize();
        int headSize = parameter.headDim();
        int queryHeads = parameter.getNumAttentionHeads();
        int keyValueHeads = parameter.getNumKeyValueHeads();
        int groupSize = parameter.keyValueGroupSize();
        double scale = 1.0 / Math.sqrt(headSize);

        // Build the shared key and value heads (with RoPE added to the keys).
        ComputationalNode[] keys = new ComputationalNode[keyValueHeads];
        ComputationalNode[] values = new ComputationalNode[keyValueHeads];
        for (int kv = 0; kv < keyValueHeads; kv++) {
            ComputationalNode key = linearLayer(input, hiddenSize, headSize, random);
            keys[kv] = this.addEdge(key, new RotaryPositionalEmbedding());
            values[kv] = linearLayer(input, hiddenSize, headSize, random);
        }

        // Build every query head and compute its attention output.
        ArrayList<ComputationalNode> headResults = new ArrayList<>();
        for (int head = 0; head < queryHeads; head++) {
            int kv = head / groupSize;
            ComputationalNode query = linearLayer(input, hiddenSize, headSize, random);
            ComputationalNode rotatedQuery = this.addEdge(query, new RotaryPositionalEmbedding());
            ComputationalNode keyTransposed = this.addEdge(keys[kv], new Transpose());
            ComputationalNode scores = this.addEdge(rotatedQuery, keyTransposed, false, false);
            ComputationalNode scaledScores = this.addEdge(scores, new MultiplyByConstant(scale));
            ComputationalNode maskedScores = this.addEdge(scaledScores, new Mask());
            ComputationalNode attentionWeights = this.addEdge(maskedScores, new Softmax());
            headResults.add(this.addEdge(attentionWeights, values[kv], false, false));
        }

        // Glue the heads back together and project to the hidden size.
        ConcatenatedNode allHeads = (ConcatenatedNode) this.concatEdges(headResults, 1);
        return linearLayer(allHeads, hiddenSize, hiddenSize, random);
    }

    /**
     * Adds the feed-forward part of a block (SwiGLU).
     *
     * <p>It computes two projections of the input. The first one goes through the
     * SiLU activation and acts as a "gate" that is multiplied element-by-element
     * with the second one. The result is projected back to the hidden size:
     * {@code out = (SiLU(x @ Wgate) * (x @ Wup)) @ Wdown}.</p>
     *
     * @param input  the (already normalised) input node.
     * @param random random generator used for the weights.
     * @return the feed-forward output node.
     */
    private ComputationalNode feedForward(ComputationalNode input, Random random) {
        int hiddenSize = parameter.getHiddenSize();
        int innerSize = parameter.getIntermediateSize();
        ComputationalNode gate = linearLayer(input, hiddenSize, innerSize, random);
        ComputationalNode activatedGate = this.addEdge(gate, new SiLU());
        ComputationalNode up = linearLayer(input, hiddenSize, innerSize, random);
        ComputationalNode gated = this.addEdge(activatedGate, up, false, true);
        return linearLayer(gated, innerSize, hiddenSize, random);
    }

    /**
     * Adds one decoder block: attention with a residual connection, then
     * feed-forward with a residual connection. We normalise before each part
     * (pre-normalisation), which is what Mistral does.
     *
     * @param input  the block input node.
     * @param random random generator used for the weights.
     * @return the block output node.
     */
    private ComputationalNode decoderBlock(ComputationalNode input, Random random) {
        ComputationalNode attentionOut = selfAttention(rmsNorm(input), random);
        ComputationalNode afterAttention = this.addAdditionEdge(input, attentionOut, false);
        ComputationalNode feedForwardOut = feedForward(rmsNorm(afterAttention), random);
        return this.addAdditionEdge(afterAttention, feedForwardOut, false);
    }

    /**
     * Reads one example into the input node and returns its correct labels.
     *
     * <p>An example is a flat list of numbers. For each token it stores the
     * embedding (hiddenSize numbers) followed by one label, so its length is
     * {@code numberOfTokens * (hiddenSize + 1)}.</p>
     *
     * @param example the flat example tensor.
     * @return the list of correct labels, one per token.
     */
    private ArrayList<Integer> readExample(Tensor example) {
        int hiddenSize = parameter.getHiddenSize();
        int numberOfTokens = example.getShape()[0] / (hiddenSize + 1);
        ArrayList<Integer> labels = new ArrayList<>();
        ArrayList<Double> embeddings = new ArrayList<>();
        int position = 0;
        for (int token = 0; token < numberOfTokens; token++) {
            for (int feature = 0; feature < hiddenSize; feature++) {
                embeddings.add(example.getValue(new int[]{position}));
                position++;
            }
            labels.add((int) example.getValue(new int[]{position}));
            position++;
        }
        this.inputNodes.get(0).setValue(new Tensor(embeddings, new int[]{numberOfTokens, hiddenSize}));
        return labels;
    }

    /**
     * Builds the graph once and then trains it for the chosen number of epochs.
     * For each example we put the embeddings into the input node, put the correct
     * labels (as one-hot vectors) into the label node, run the forward pass and
     * then back-propagation, which updates the weights.
     *
     * @param trainSet the training examples.
     */
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        Random random = new Random(parameter.getSeed());
        int hiddenSize = parameter.getHiddenSize();
        int numberOfClasses = parameter.getClassLabelSize();

        // Build the graph: input -> decoder blocks -> final norm -> head -> softmax.
        ComputationalNode input = new MultiplicationNode(false, false);
        this.inputNodes.add(input);
        ComputationalNode hidden = input;
        for (int layer = 0; layer < parameter.getNumLayers(); layer++) {
            hidden = decoderBlock(hidden, random);
        }
        ComputationalNode normalised = rmsNorm(hidden);
        ComputationalNode scores = linearLayer(normalised, hiddenSize, numberOfClasses, random);
        this.outputNode = this.addEdge(scores, new Softmax());

        // Attach the place where we will put the correct labels and the loss.
        ComputationalNode labelNode = new ComputationalNode();
        this.inputNodes.add(labelNode);
        this.addLoss(labelNode);

        // Train.
        for (int epoch = 0; epoch < parameter.getEpoch(); epoch++) {
            this.shuffle(trainSet, random);
            for (Tensor example : trainSet) {
                ArrayList<Integer> labels = readExample(example);
                ArrayList<Double> oneHotLabels = new ArrayList<>();
                for (Integer label : labels) {
                    for (int c = 0; c < numberOfClasses; c++) {
                        oneHotLabels.add(c == label ? 1.0 : 0.0);
                    }
                }
                labelNode.setValue(new Tensor(oneHotLabels, new int[]{labels.size(), numberOfClasses}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }

    /**
     * Turns the softmax output into one predicted label per token by picking the
     * class with the highest probability.
     *
     * @return the predicted labels (as doubles), one per token.
     */
    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> predictions = new ArrayList<>();
        Tensor output = this.outputNode.getValue();
        for (int token = 0; token < output.getShape()[0]; token++) {
            int bestClass = -1;
            double bestProbability = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < output.getShape()[1]; c++) {
                double probability = output.getValue(new int[]{token, c});
                if (probability > bestProbability) {
                    bestProbability = probability;
                    bestClass = c;
                }
            }
            predictions.add((double) bestClass);
        }
        return predictions;
    }

    /**
     * Runs the model on the test set and returns the accuracy (how many tokens we
     * labelled correctly, divided by the total number of tokens).
     *
     * @param testSet the test examples (same format as the training examples).
     * @return the accuracy on the test set.
     */
    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int correct = 0;
        int total = 0;
        for (Tensor example : testSet) {
            ArrayList<Integer> correctLabels = readExample(example);
            ArrayList<Double> predictedLabels = this.predict();
            for (int token = 0; token < correctLabels.size(); token++) {
                if (correctLabels.get(token).equals(predictedLabels.get(token).intValue())) {
                    correct++;
                }
                total++;
            }
        }
        return new ClassificationPerformance((correct + 0.0) / total);
    }
}
