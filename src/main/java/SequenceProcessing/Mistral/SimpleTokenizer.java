package SequenceProcessing.Mistral;

/**
 * Tiny byte/character tokenizer for project... Mistral uses a SentencePiece tokenizer.
 */
public class SimpleTokenizer {
    private final int vocabSize;

    public SimpleTokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int[] encode(String text) {
        int[] tokens = new int[Math.min(text.length(), 256)];
        for (int i = 0; i < tokens.length; i++) tokens[i] = text.charAt(i) % vocabSize;
        return tokens;
    }

    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            if (token >= 32 && token <= 126) sb.append((char) token);
            else sb.append('?');
        }
        return sb.toString();
    }
}
