package SequenceProcessing.Functions;

import ComputationalGraph.Function;
import Math.Tensor;

import java.io.Serializable;

public class Transpose implements Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        return tensor.transpose(new int[]{0, 2, 1});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.transpose(new int[]{0, 2, 1});
    }
}
