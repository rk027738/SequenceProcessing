# Mistral Model Implementation (Computational Graph)

## Overview
This project is a simple implementation of a Mistral-style, decoder-only
language model written in Java. The model is fixed and is built **on top of the
`ComputationalGraph` library** : every operation is
expressed as a differentiable edge in a graph, so the library performs the
forward pass, automatic differentiation (back-propagation) and the optimizer
update.

### Implemented Mistral Features
- Root-Mean-Square normalization (RMSNorm) with a learnable gain
- Rotary Positional Embeddings (RoPE) applied to queries and keys
- Grouped-Query Attention (GQA)
- Causally masked self-attention
- SwiGLU feed-forward network
- Residual connections (pre-normalisation blocks)
- A softmax language-modeling head trained with cross-entropy

### Architecture
```
Input embeddings -> [Decoder Block] x N -> Final RMSNorm -> LM Head -> Softmax
```
Each decoder block (pre-norm + residuals):
```
RMSNorm -> GQA attention -> add residual -> RMSNorm -> SwiGLU -> add residual
```
Because the graph operates on whole `[sequenceLength, hiddenSize]` matrices, a
single graph handles sequences of any length.

## Running the Project

### Requirements
- Java 8+
- Maven 3.8+

### Compile
```
mvn clean compile
```

### Test
```
mvn test -Dtest=MistralModelTest
```

### Run the demo
```
SequenceProcessing.Mistral.MistralDemo
```
The demo builds a small model, trains it on a synthetic dataset and prints the
resulting accuracy.

## Important Classes
- `Mistral/MistralModel.java` &ndash; builds the whole network as a
  `ComputationalGraph` (RMSNorm, RoPE, GQA, SwiGLU, residuals, LM head) and
  implements `train`/`test`.
- `Parameters/MistralParameter.java` &ndash; the model hyper-parameters; extends
  `NeuralNetworkParameter` so the model plugs into the shared optimizer/loss
  infrastructure.
- `Functions/RotaryPositionalEmbedding.java` &ndash; the RoPE graph function
  (with the matching gradient).
- RMSNorm, attention scaling, causal masking and SwiGLU are assembled from the
  reusable graph functions in `SequenceProcessing/Functions` (`Variance`,
  `SquareRoot`, `Inverse`, `Mask`, `MultiplyByConstant`, `Transpose`) together
  with the library's `SiLU` and `Softmax`.

## Data format
Each training/test instance is a flat `Math.Tensor` that stores, for every
token, its `hiddenSize` embedding values followed by a single integer class
label, i.e. its length is `sequenceLength * (hiddenSize + 1)`.

## Difference from other models
This model uses RMSNorm, RoPE, GQA and SwiGLU, which are associated with modern
Mistral architectures rather than older GPT-style implementations.

## Limitations
This is an educational implementation and does not include Mixtral sparse
Mixture-of-Experts, KV caching or production-scale training.
