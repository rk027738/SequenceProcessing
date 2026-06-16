import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.AdamW;
import Math.Tensor;
import SequenceProcessing.Functions.RotaryPositionalEmbedding;
import SequenceProcessing.Mistral.MistralModel;
import SequenceProcessing.Parameters.MistralParameter;
import Classification.Performance.ClassificationPerformance;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Tests for the graph-based {@link MistralModel}.
 *
 * <p>The tests check two things. First, that the rotary position embedding
 * behaves like a rotation should (it keeps the shape, does nothing at position 0,
 * and its gradient undoes the rotation). Second, that the model actually works
 * end to end: it builds, trains, gives back a sensible accuracy, learns better
 * than random guessing, and gives the same result twice when we use the same
 * seed.</p>
 *
 * <p>Because the model needs data, the helper {@link #makeData} creates small fake
 * sentences where each token's label depends on its embedding, so there is a real
 * pattern to learn.</p>
 */
public class MistralModelTest {

    private static final int HIDDEN_SIZE = 16;
    private static final int NUMBER_OF_CLASSES = 4;

    /**
     * Makes a small fake dataset. Each example is one flat list of numbers: for
     * every token we store its embedding followed by its label.
     *
     * @param howMany how many examples to make.
     * @param random  the random generator to use.
     * @return the list of examples.
     */
    private ArrayList<Tensor> makeData(int howMany, Random random) {
        ArrayList<Tensor> data = new ArrayList<>();
        for (int i = 0; i < howMany; i++) {
            int numberOfTokens = 3 + random.nextInt(3);
            ArrayList<Double> numbers = new ArrayList<>();
            for (int token = 0; token < numberOfTokens; token++) {
                double sum = 0.0;
                for (int feature = 0; feature < HIDDEN_SIZE; feature++) {
                    double value = random.nextGaussian();
                    numbers.add(value);
                    sum += value;
                }
                numbers.add((double) Math.floorMod((int) Math.round(sum), NUMBER_OF_CLASSES));
            }
            data.add(new Tensor(numbers, new int[]{numbers.size()}));
        }
        return data;
    }

    /**
     * Builds a small set of model settings used by several tests.
     *
     * @return the model settings.
     */
    private MistralParameter smallSettings() {
        return new MistralParameter(
                1, 5,
                new AdamW(0.01, 0.99, 0.9, 0.999, 1e-8, 0.01),
                new RandomInitialization(),
                new CrossEntropyLoss(),
                HIDDEN_SIZE, 2, 4, 2, 32, NUMBER_OF_CLASSES, 1e-6);
    }

    /**
     * Counts how many tokens there are in a whole dataset (every example can have
     * a different number of tokens).
     *
     * @param data the dataset.
     * @return the total number of tokens.
     */
    private int countTokens(ArrayList<Tensor> data) {
        int total = 0;
        for (Tensor example : data) {
            total += example.getShape()[0] / (HIDDEN_SIZE + 1);
        }
        return total;
    }

    /**
     * Reads the correct label of every token out of one flat example, so we can
     * print the data and see what it looks like.
     *
     * @param example one flat example.
     * @return the list of labels, one per token.
     */
    private ArrayList<Integer> labelsOf(Tensor example) {
        int numberOfTokens = example.getShape()[0] / (HIDDEN_SIZE + 1);
        ArrayList<Integer> labels = new ArrayList<>();
        for (int token = 0; token < numberOfTokens; token++) {
            int labelPosition = token * (HIDDEN_SIZE + 1) + HIDDEN_SIZE;
            labels.add((int) example.getValue(new int[]{labelPosition}));
        }
        return labels;
    }

    /**
     * The rotary embedding should keep the shape, and at position 0 it should
     * change nothing (the rotation angle is zero there).
     */
    @Test
    public void rotaryKeepsShapeAndDoesNothingAtPositionZero() {
        RotaryPositionalEmbedding rotary = new RotaryPositionalEmbedding();
        ArrayList<Double> numbers = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            numbers.add((double) (i + 1));
        }
        Tensor input = new Tensor(numbers, new int[]{2, 4});

        Tensor output = rotary.calculate(input);

        assertArrayEquals(input.getShape(), output.getShape());
        for (int feature = 0; feature < 4; feature++) {
            assertEquals(input.getValue(new int[]{0, feature}), output.getValue(new int[]{0, feature}), 1e-9);
        }
    }

    /**
     * Rotating a vector and then rotating it back (the gradient does the opposite
     * rotation) should give the original vector again.
     */
    @Test
    public void rotaryGradientUndoesTheRotation() {
        RotaryPositionalEmbedding rotary = new RotaryPositionalEmbedding();
        ArrayList<Double> numbers = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            numbers.add(0.5 * (i + 1));
        }
        Tensor input = new Tensor(numbers, new int[]{2, 4});

        Tensor rotated = rotary.calculate(input);
        Tensor backAgain = rotary.derivative(rotated, rotated);

        for (int token = 0; token < 2; token++) {
            for (int feature = 0; feature < 4; feature++) {
                assertEquals(input.getValue(new int[]{token, feature}), backAgain.getValue(new int[]{token, feature}), 1e-9);
            }
        }
    }

    /**
     * The head size must be even for rotary embeddings, so a setting that gives an
     * odd head size should be refused.
     */
    @Test(expected = IllegalArgumentException.class)
    public void oddHeadSizeIsNotAllowed() {
        new MistralParameter(1, 1, new AdamW(0.01, 0.99, 0.9, 0.999, 1e-8, 0.01),
                new RandomInitialization(), new CrossEntropyLoss(),
                12, 1, 4, 2, 16, NUMBER_OF_CLASSES, 1e-6);
    }

    /**
     * After training, the accuracy should be a normal number between 0 and 1.
     * This checks the whole pipeline (build, forward, back-propagation, update)
     * runs without errors.
     */
    @Test
    public void modelSuccessfullyRuns() {
        Random random = new Random(7L);
        MistralModel model = new MistralModel(smallSettings());
        ArrayList<Tensor> trainSet = makeData(24, random);
        ArrayList<Tensor> testSet = makeData(8, random);

        System.out.println("=== modelSuccessfullyRuns ===");
        System.out.println("Train examples: " + trainSet.size() + " (" + countTokens(trainSet) + " tokens)");
        System.out.println("Test examples : " + testSet.size() + " (" + countTokens(testSet) + " tokens)");
        Tensor firstExample = testSet.get(0);
        int firstExampleTokens = firstExample.getShape()[0] / (HIDDEN_SIZE + 1);
        System.out.println("First test example -> " + firstExampleTokens + " tokens, gold labels: " + labelsOf(firstExample));

        model.train(trainSet);
        ClassificationPerformance result = model.test(testSet);

        System.out.printf("Test accuracy: %.4f%n", result.getAccuracy());
        System.out.println();

        assertTrue(result.getAccuracy() >= 0.0);
        assertTrue(result.getAccuracy() <= 1.0);
    }

    /**
     * The model should do better than random guessing on the data it was trained
     * on. With four equally likely classes, random guessing is about 0.25.
     */
    @Test
    public void modelLearnsBetterThanRandom() {
        Random random = new Random(123L);
        MistralModel model = new MistralModel(smallSettings());
        ArrayList<Tensor> trainSet = makeData(40, random);

        double randomBaseline = 1.0 / NUMBER_OF_CLASSES;
        System.out.println("=== modelLearnsBetterThanRandom ===");
        System.out.println("Training examples: " + trainSet.size() + " (" + countTokens(trainSet) + " tokens)");
        System.out.printf("Random guessing baseline (%d classes): %.4f%n", NUMBER_OF_CLASSES, randomBaseline);

        model.train(trainSet);
        ClassificationPerformance result = model.test(trainSet);

        System.out.printf("Training accuracy after %d epochs: %.4f%n", 5, result.getAccuracy());
        System.out.printf("Improvement over random: %+.4f%n", result.getAccuracy() - randomBaseline);
        System.out.println();

        assertTrue("Expected the model to beat random guessing", result.getAccuracy() > 0.25);
    }

    /**
     * Using the same seed and the same data should give exactly the same result
     * every time. Each model gets its own copy of the training list because
     * training shuffles the list in place.
     */
    @Test
    public void RandomSeedsGiveSameResult() {
        ArrayList<Tensor> trainSet = makeData(20, new Random(99L));
        ArrayList<Tensor> testSet = makeData(8, new Random(100L));

        MistralModel firstModel = new MistralModel(smallSettings());
        firstModel.train(new ArrayList<>(trainSet));
        double firstAccuracy = firstModel.test(testSet).getAccuracy();

        MistralModel secondModel = new MistralModel(smallSettings());
        secondModel.train(new ArrayList<>(trainSet));
        double secondAccuracy = secondModel.test(testSet).getAccuracy();

        assertEquals(firstAccuracy, secondAccuracy, 1e-12);
    }
}
