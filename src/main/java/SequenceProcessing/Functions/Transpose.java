package SequenceProcessing.Functions;

import Math.Tensor;

import java.io.Serializable;

public class Transpose implements ComputationalGraph.Function.Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        return tensor.transpose(new int[]{1, 0});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.transpose(new int[]{1, 0});
    }
}
