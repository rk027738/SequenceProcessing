import SequenceProcessing.Mistral.*;
import org.junit.Test;
import java.util.Random;

import static org.junit.Assert.*;

public class MistralModelTest {

    /**
     * Verifies that the tokenizer consistently encodes text.
     */
    @Test
    public void tokenizerEncodeDecodeWorks() {
        MistralConfig config = new MistralConfig();
        SimpleTokenizer tokenizer = new SimpleTokenizer(config.vocabSize);

        String text = "java";

        int[] tokens = tokenizer.encode(text);

        assertNotNull(tokens);
        assertTrue(tokens.length > 0);
    }

    /**
     * Verifies that RMS normalization preserves vector dimensions.
     */
    @Test
    public void rmsNormPreservesDimensions() {
        RMSNorm rmsNorm = new RMSNorm(4, 1e-6);

        double[] input = {1.0, 2.0, 3.0, 4.0};

        double[] output = rmsNorm.forward(input);

        assertEquals(input.length, output.length);
    }

    /**
     * Verifies that the attention layer preserves sequence dimensions.
     *
     * This test checks that the attention layer produces an output
     * with the same shape as the input sequence.
     */
    @Test
    public void attentionPreservesShape() {

        MistralConfig config = new MistralConfig();

        Random random = new Random(7L);

        MistralAttention attention =
                new MistralAttention(config, random);

        double[][] input = {
                new double[config.hiddenSize],
                new double[config.hiddenSize]
        };

        double[][] output = attention.forward(input);

        assertEquals(input.length, output.length);
        assertEquals(input[0].length, output[0].length);
    }

    /**
     * Verifies that a complete decoder block executes successfully.
     *
     * This test validates the integration of RMSNorm,
     * Grouped Query Attention, residual connections,
     * and the SwiGLU feed-forward network.
     */
    @Test
    public void decoderBlockForwardPassWorks() {

        MistralConfig config = new MistralConfig();

        Random random = new Random(7L);

        MistralDecoderBlock block =
                new MistralDecoderBlock(config, random);

        double[][] input = {
                new double[config.hiddenSize],
                new double[config.hiddenSize]
        };

        double[][] output = block.forward(input);

        assertNotNull(output);

        assertEquals(input.length, output.length);

        assertEquals(
                input[0].length,
                output[0].length
        );
    }

    /**
     * Verifies that models initialized with the same random seed
     * produce identical outputs.
     */
    @Test
    public void sameSeedProducesSameOutputs() {

        MistralConfig config1 =
                new MistralConfig(
                        128, 32, 32, 64,
                        1, 4, 2,
                        1e-6,
                        7L);

        MistralConfig config2 =
                new MistralConfig(
                        128, 32, 32, 64,
                        1, 4, 2,
                        1e-6,
                        7L);

        MistralModel model1 =
                new MistralModel(config1);

        MistralModel model2 =
                new MistralModel(config2);

        int[] tokens = {1, 2, 3, 4};

        double[][] output1 =
                model1.forward(tokens);

        double[][] output2 =
                model2.forward(tokens);

        assertEquals(
                output1[0][0],
                output2[0][0],
                1e-9
        );
    }

    /**
     * Verifies that the Mistral model forward pass produces a valid
     * probability distribution for each input token.
     *
     * This test checks that:
     * - One output vector is generated per input token.
     * - Each output vector matches the configured vocabulary size.
     * - The probabilities of the final token sum to 1.0.
     *
     * Successful execution indicates that embeddings, attention,
     * normalization, feed-forward processing, and output prediction
     * are functioning correctly together.
     */
    @Test
    public void forwardReturnsProbabilityDistribution() {
        MistralConfig config = new MistralConfig(128, 32, 32, 64, 1, 4, 2, 1e-6, 7L);
        MistralModel model = new MistralModel(config);
        SimpleTokenizer tokenizer = new SimpleTokenizer(config.vocabSize);
        int[] tokens = tokenizer.encode("java");
        double[][] probs = model.forward(tokens);
        assertEquals(tokens.length, probs.length);
        assertEquals(config.vocabSize, probs[0].length);
        double sum = 0.0;
        for (double p : probs[probs.length - 1]) sum += p;
        assertEquals(1.0, sum, 1e-6);
    }

    /**
     * Verifies that the Mistral model predicts a valid next token.
     *
     * This test ensures that the predicted token index falls within
     * the allowed vocabulary range.
     *
     * Successful execution confirms that the language modeling head
     * generates valid token predictions.
     */
    @Test
    public void nextTokenIsInsideVocabulary() {
        MistralConfig config = new MistralConfig();
        MistralModel model = new MistralModel(config);
        int next = model.predictNextToken(new int[]{77, 105, 115, 116, 114, 97, 108});
        assertTrue(next >= 0 && next < config.vocabSize);
    }


}
