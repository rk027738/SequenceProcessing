package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Rotary Positional Embedding (RoPE) as used by the Mistral attention layer.
 * <p>
 * RoPE injects positional information by rotating consecutive pairs of features
 * of a query/key vector by an angle that depends on the token position. For a
 * vector of dimension {@code d} and position {@code p}, the pair
 * {@code (x[2i], x[2i+1])} is rotated by {@code angle = p * 10000^(-2i/d)}:
 * <pre>
 *     x'[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
 *     x'[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)
 * </pre>
 * The function operates on a {@code [sequenceLength, headDim]} tensor where the
 * row index is interpreted as the absolute token position. {@code headDim} must
 * be even.
 * <p>
 * Because each rotation is an orthogonal (length-preserving) linear map, its
 * gradient is simply the rotation by the opposite angle, which is what
 * {@link #derivative(Tensor, Tensor)} applies to the incoming gradient.
 */
public class RotaryPositionalEmbedding implements Function, Serializable {

    /** Base used for the geometric progression of rotation frequencies. */
    private static final double BASE = 10000.0;

    /**
     * Applies the rotary embedding to every row (token position) of the tensor.
     *
     * @param tensor A {@code [sequenceLength, headDim]} tensor of query or key features.
     * @return A new tensor of the same shape with positional rotations applied.
     */
    @Override
    public Tensor calculate(Tensor tensor) {
        int sequenceLength = tensor.getShape()[0];
        int headDim = tensor.getShape()[1];
        ArrayList<Double> values = new ArrayList<>();
        for (int position = 0; position < sequenceLength; position++) {
            for (int i = 0; i < headDim; i += 2) {
                double angle = position * Math.pow(BASE, -((double) i / headDim));
                double cos = Math.cos(angle);
                double sin = Math.sin(angle);
                double even = tensor.getValue(new int[]{position, i});
                double odd = tensor.getValue(new int[]{position, i + 1});
                values.add(even * cos - odd * sin);
                values.add(even * sin + odd * cos);
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Propagates the gradient back through the rotation by rotating the incoming
     * gradient with the transposed (opposite-angle) rotation matrix.
     *
     * @param value    The output produced by {@link #calculate(Tensor)} (used only for its shape).
     * @param backward The gradient flowing back from the next node.
     * @return The gradient with respect to the inputs of this function.
     */
    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        int sequenceLength = backward.getShape()[0];
        int headDim = backward.getShape()[1];
        ArrayList<Double> values = new ArrayList<>();
        for (int position = 0; position < sequenceLength; position++) {
            for (int i = 0; i < headDim; i += 2) {
                double angle = position * Math.pow(BASE, -((double) i / headDim));
                double cos = Math.cos(angle);
                double sin = Math.sin(angle);
                double gradEven = backward.getValue(new int[]{position, i});
                double gradOdd = backward.getValue(new int[]{position, i + 1});
                values.add(gradEven * cos + gradOdd * sin);
                values.add(-gradEven * sin + gradOdd * cos);
            }
        }
        return new Tensor(values, backward.getShape());
    }
}
