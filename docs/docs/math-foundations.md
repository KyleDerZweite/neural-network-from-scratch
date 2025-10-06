---
title: Mathematical Foundations
sidebar_position: 3
description: Linear algebra and calculus underpinning nn-core.
---

# Mathematical Foundations

This module collects the algebra and calculus used throughout `nn-core`. Symbols mirror those in the source so you can jump between code and equations effortlessly.

## Notation

- $x^{(\ell)}$ — activation vector at layer $\ell$.
- $W^{(\ell)}$ — weight matrix connecting layer $\ell - 1$ to $\ell$.
- $b^{(\ell)}$ — bias vector of layer $\ell$.
- $f^{(\ell)}$ — activation function applied at layer $\ell$.
- $\hat{y}$ — model prediction.
- $y$ — ground-truth label.

## Matrix Algebra

Let $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$. `nn-core` implements the following primitives:

- **Matrix multiplication**:
  $$
  (AB)_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
  $$
  Encoded by [`Matrix::mul`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs#L37) and used to propagate activations forward.

- **Element-wise (Hadamard) product**:
  $$
  (A \odot B)_{ij} = A_{ij} B_{ij}
  $$
  Exposed as [`Matrix::dot`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs#L91). Essential for combining error terms with activation derivatives.

- **Transpose-aware multiplication**:
  $$
  A B^{\top} \quad\text{and}\quad A^{\top} B
  $$
  Implemented via [`mul_transpose`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs#L59) and [`transpose_mul`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs#L74) to avoid creating explicit transpose matrices during backpropagation.

## Activation Function

`nn-core` adopts the sigmoid non-linearity for hidden units, defined in [`activation.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/activation.rs):

- **Sigmoid**:
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$

- **Derivative (input form)**:
  $$
  \sigma'(x) = \sigma(x) \bigl(1 - \sigma(x)\bigr)
  $$

- **Derivative (output form)**:
  $$
  \sigma'(y) = y (1 - y)
  $$

The output-form derivative is particularly convenient because the network already caches $y = \sigma(x)$ during the forward pass.

## Feed-Forward Equations

For each layer $\ell = 1, \dotsc, L$:

1. Affine transform:
   $$
   z^{(\ell)} = W^{(\ell)} x^{(\ell-1)} + b^{(\ell)}
   $$
2. Activation:
   $$
   x^{(\ell)} = f^{(\ell)}\!\left(z^{(\ell)}\right)
   $$

These equations are applied sequentially in [`NeuralNetwork::train`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/network.rs#L33) and [`NeuralNetwork::predict`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/network.rs#L108).

## Loss Function

For scalar outputs, `nn-core` minimises mean squared error (MSE):
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \bigl(\hat{y}_i - y_i\bigr)^2
$$

Gradients with respect to the prediction are simply $\partial \mathcal{L} / \partial \hat{y} = 2(\hat{y} - y)/N$. Because batches are size one in the base implementation, the factor of $N$ reduces to 1 during training.

## Backpropagation

### Output Layer

Define the error $E = \hat{y} - y$. For a sigmoid output layer:
$$
\delta^{(L)} = E \odot \sigma'(\hat{y})
$$

If the output layer is linear, the derivative becomes $\delta^{(L)} = E$.

### Hidden Layers

Working backwards for $\ell = L-1, \dotsc, 1$:
$$
\delta^{(\ell)} = \left(W^{(\ell+1)}\right)^{\top} \delta^{(\ell+1)} \odot f'\bigl(x^{(\ell)}\bigr)
$$

### Parameter Gradients

- **Weights**:
  $$
  \frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)} \left(x^{(\ell-1)}\right)^{\top}
  $$
- **Biases**:
  $$
  \frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}
  $$

These gradients feed directly into stochastic gradient descent.

## Optimisation Step

Let the learning rate be $\eta$. We update each parameter pair as
$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta \, \frac{\partial \mathcal{L}}{\partial W^{(\ell)}}
$$
$$
b^{(\ell)} \leftarrow b^{(\ell)} - \eta \, \frac{\partial \mathcal{L}}{\partial b^{(\ell)}}
$$

`nn-core` applies this rule after every sample, performing true stochastic gradient descent.

With the mathematical scaffolding in place, the next section shows how each formula manifests in the Rust source.
