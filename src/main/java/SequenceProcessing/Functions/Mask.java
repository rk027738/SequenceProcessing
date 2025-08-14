package SequenceProcessing.Functions;

import ComputationalGraph.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Mask implements Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[1]; i++) {
            for (int j = 0; j < tensor.getShape()[2]; j++) {
                if (j > i) {
                    values.add(Double.NEGATIVE_INFINITY);
                } else {
                    values.add(tensor.getValue(new int[]{0, i, j}));
                }
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[1]; i++) {
            for (int j = 0; j < tensor.getShape()[2]; j++) {
                values.add(1.0);
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
