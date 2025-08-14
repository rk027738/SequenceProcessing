package SequenceProcessing.Parameters;

import Classification.Parameter.LinearPerceptronParameter;
import ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

public class TransformerParameter extends LinearPerceptronParameter implements Serializable {

    private final int L;
    private final int N;
    private final int V;
    private final double epsilon;
    private final ArrayList<Integer> inputHiddenLayers;
    private final ArrayList<Integer> outputHiddenLayers;
    private final ArrayList<Function> inputFunctions;
    private final ArrayList<Function> outputFunctions;

    public TransformerParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, int wordEmbeddingLength, int multiHeadAttentionLength, int vocabularyLength, double epsilon, ArrayList<Integer> inputHiddenLayers, ArrayList<Integer> outputHiddenLayers, ArrayList<Function> inputActivationFunctions, ArrayList<Function> outputActivationFunctions) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
        this.L = wordEmbeddingLength;
        this.N = multiHeadAttentionLength;
        this.V = vocabularyLength;
        this.epsilon = epsilon;
        this.inputHiddenLayers = inputHiddenLayers;
        this.outputHiddenLayers = outputHiddenLayers;
        this.inputFunctions = inputActivationFunctions;
        this.outputFunctions = outputActivationFunctions;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public int getDk() {
        return L / N;
    }

    public int getL() {
        return L;
    }

    public int getN() {
        return N;
    }

    public int getV() {
        return V;
    }

    public int getInputHiddenLayer(int index) {
        return inputHiddenLayers.get(index);
    }

    public int getOutputHiddenLayer(int index) {
        return outputHiddenLayers.get(index);
    }

    public Function getInputActivationFunction(int index) {
        return inputFunctions.get(index);
    }

    public Function getOutputActivationFunction(int index) {
        return outputFunctions.get(index);
    }

    public int getInputSize() {
        return inputHiddenLayers.size();
    }

    public int getOutputSize() {
        return outputHiddenLayers.size();
    }

    public void setLearningRate() {
        this.learningRate *= etaDecrease;
    }
}
