# Simple Mistral Model Implementation Project

## Overview
This project is a simplified educational implementation of a Mistral-style decoder-only language model written in Java.

### Implemented Mistral Features
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Causal Masked Self-Attention
- SwiGLU Feed Forward Network
- Residual Connections
- Decoder-Only Next Token Prediction

### Architecture
Input Tokens -> Embeddings -> Decoder Blocks -> Final RMSNorm -> LM Head -> Softmax

Each Decoder Block:
RMSNorm -> GQA Attention -> Residual -> RMSNorm -> SwiGLU -> Residual

## Running the Project

### Requirements
- Java 17+
- Maven 3.8+

### Compile
mvn clean compile

### Run Demo
Run:
SequenceProcessing.Mistral.MistralDemo

or

mvn exec:java

## Important Classes
MistralModel.java - Main model orchestration

MistralDecoderBlock.java - Transformer decoder block

MistralAttention.java - GQA attention with RoPE

SwiGLUFeedForward.java - Feed-forward network

RMSNorm.java - Root Mean Square Normalization

RotaryEmbedding.java - Positional encoding

## Difference from other models
This project uses RMSNorm, RoPE, GQA and SwiGLU, which are associated with modern Mistral architectures rather than older GPT-style implementations.

## Limitation
This is a very basic implementation and does not include Mixtral Sparse Mixture-of-Experts, KV caching, or production-scale training.
