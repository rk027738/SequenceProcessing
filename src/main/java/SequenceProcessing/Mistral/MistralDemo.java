package SequenceProcessing.Mistral;

import java.util.Arrays;

/** Command-line demo */
public class MistralDemo {
    public static void main(String[] args) {
        MistralConfig config = new MistralConfig();
        SimpleTokenizer tokenizer = new SimpleTokenizer(config.vocabSize);
        MistralModel model = new MistralModel(config);

        String prompt = args.length == 0 ? "Mistral in Java" : String.join(" ", args);
        int[] tokens = tokenizer.encode(prompt);
        int nextToken = model.predictNextToken(tokens);
        double[][] probabilities = model.forward(tokens);

        System.out.println("Prompt: " + prompt);
        System.out.println("Input token ids: " + Arrays.toString(tokens));
        System.out.println("Predicted next token id: " + nextToken);
        System.out.printf("Probability of predicted token: %.6f%n", probabilities[probabilities.length - 1][nextToken]);
        System.out.println("Decoded predicted token: " + tokenizer.decode(new int[]{nextToken}));
    }
}
