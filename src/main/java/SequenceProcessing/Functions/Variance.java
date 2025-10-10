package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Variance implements Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> variances = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            double total = 0.0;
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                total += Math.pow(tensor.getValue(new int[]{i, j}), 2);
            }
            variances.add(total / tensor.getShape()[1]);
        }
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values.add(variances.get(i));
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                values.add(2.0 * Math.sqrt(tensor.getShape()[1] * tensor.getValue(new int[]{i, j})) / tensor.getShape()[1]);
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}
