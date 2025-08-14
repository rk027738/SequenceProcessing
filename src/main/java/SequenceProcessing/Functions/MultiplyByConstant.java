package SequenceProcessing.Functions;

import ComputationalGraph.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class MultiplyByConstant implements Function, Serializable {

    private final double constant;

    public MultiplyByConstant(double constant) {
        this.constant = constant;
    }

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Double> tensorValues = (ArrayList<Double>) tensor.getData();
        for (double val : tensorValues) {
            double newVal = constant * val;
            values.add(newVal);
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor tensor, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[1]; i++) {
            for (int j = 0; j < tensor.getShape()[2]; j++) {
                values.add(constant);
            }
        }
        return backward.hadamardProduct(new Tensor(values, tensor.getShape()));
    }
}