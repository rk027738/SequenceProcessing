import SequenceProcessing.Mistral.*;
import org.junit.Test;

import static org.junit.Assert.*;

public class MistralModelTest {
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

    @Test
    public void nextTokenIsInsideVocabulary() {
        MistralConfig config = new MistralConfig();
        MistralModel model = new MistralModel(config);
        int next = model.predictNextToken(new int[]{77, 105, 115, 116, 114, 97, 108});
        assertTrue(next >= 0 && next < config.vocabSize);
    }
}
