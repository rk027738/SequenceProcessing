package SequenceProcessing.Functions;

import ComputationalGraph.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Inverse implements Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values.add(1.0 / tensor.getValue(new int[]{i, j}));
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values.add(-Math.pow(tensor.getValue(new int[]{i, j}), 2));
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
