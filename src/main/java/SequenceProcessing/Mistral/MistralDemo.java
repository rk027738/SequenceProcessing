package SequenceProcessing.Mistral;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.AdamW;
import Math.Tensor;
import SequenceProcessing.Parameters.MistralParameter;

import java.util.ArrayList;
import java.util.Random;

/**
 * A tiny program that shows how to use {@link MistralModel}.
 *
 * <p>It makes up a small, easy dataset, trains the model on it and prints how
 * accurate the trained model is. The task is made simple on purpose so the whole
 * thing runs in a few seconds: every token gets a random embedding and a label
 * that depends on that embedding, so there is a real pattern for the model to
 * learn.</p>
 */
public class MistralDemo {

    /** How many numbers describe each token. */
    private static final int HIDDEN_SIZE = 16;
    /** How many different labels a token can have. */
    private static final int NUMBER_OF_CLASSES = 4;

    /**
     * Makes up a list of fake sentences for the demo.
     *
     * <p>Each sentence is stored as one long list of numbers. For every token we
     * add its embedding (HIDDEN_SIZE numbers) and then its label.</p>
     *
     * @param howMany how many sentences to make.
     * @param random  the random generator to use.
     * @return the list of examples.
     */
    private static ArrayList<Tensor> makeFakeData(int howMany, Random random) {
        ArrayList<Tensor> data = new ArrayList<>();
        for (int i = 0; i < howMany; i++) {
            int numberOfTokens = 3 + random.nextInt(4);
            ArrayList<Double> numbers = new ArrayList<>();
            for (int token = 0; token < numberOfTokens; token++) {
                double sum = 0.0;
                for (int feature = 0; feature < HIDDEN_SIZE; feature++) {
                    double value = random.nextGaussian();
                    numbers.add(value);
                    sum += value;
                }
                int label = Math.floorMod((int) Math.round(sum), NUMBER_OF_CLASSES);
                numbers.add((double) label);
            }
            data.add(new Tensor(numbers, new int[]{numbers.size()}));
        }
        return data;
    }

    /**
     * Runs the demo.
     *
     * @param args not used.
     */
    public static void main(String[] args) {
        Random random = new Random(42L);

        // Settings of the model. AdamW is the optimizer and cross entropy is the loss.
        MistralParameter settings = new MistralParameter(
                1, 20,
                new AdamW(0.01, 0.99, 0.9, 0.999, 1e-8, 0.01),
                new RandomInitialization(),
                new CrossEntropyLoss(),
                HIDDEN_SIZE, 2, 4, 2, 32, NUMBER_OF_CLASSES, 1e-6);

        ArrayList<Tensor> trainSet = makeFakeData(64, random);
        ArrayList<Tensor> testSet = makeFakeData(16, random);

        MistralModel model = new MistralModel(settings);
        System.out.println("Training a graph-based Mistral model ...");
        model.train(trainSet);

        ClassificationPerformance result = model.test(testSet);
        System.out.printf("Test accuracy: %.4f%n", result.getAccuracy());
    }
}
