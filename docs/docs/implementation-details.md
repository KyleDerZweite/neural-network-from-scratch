---
title: Implementation Walkthrough
sidebar_position: 4
description: Mapping mathematical formulas to the nn-core Rust source.
---

# Implementation Walkthrough

This section mirrors the mathematical foundations with concrete Rust snippets. Each subsystem points to the exact files and functions inside `nn-core`.

## Matrix Layer

File: [`crates/nn-core/src/matrix.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs)

Key responsibilities:

- **Construction**: `Matrix::new` validates dimensions, while `Matrix::from_vec` converts column vectors into 2D form.
- **Affine algebra**: `add`, `subtract`, and `mul` implement $A \pm B$ and $AB$ respectively.
- **Transpose helpers**: `mul_transpose` and `transpose_mul` compute $\delta x^{\top}$ and $W^{\top} \delta$ without materialising large intermediates.
- **Element-wise ops**: `map` runs arbitrary closures across entries; `dot` performs Hadamard multiplication.

All routines include `profile_scope!` markers so you can enable the `profiling` feature and observe timing breakdowns.

## Activation Layer

File: [`crates/nn-core/src/activation.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/activation.rs)

Implements:

- `sigmoid(x)`
- `sigmoid_derivative(x)` — derivative using the pre-activation value.
- `sigmoid_derivative_from_output(output)` — derivative using the already-computed activation.

These functions are pure and unit-test friendly, making it straightforward to swap in other nonlinearities.

## Layer Abstraction

File: [`crates/nn-core/src/layer.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/layer.rs)

`Layer::new` seeds weights and biases using a shared `SmallRng`. A consistent offset per index prevents neurons from starting identically, avoiding the symmetry trap common in dense networks. The activation is stored as a string, allowing simple branching for sigmoid versus linear during the forward pass.

## Forward Pass

Snippet from [`NeuralNetwork::train`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/network.rs#L37):

```rust
for layer in &self.layers {
    let z = layer.weights.mul(activations.last().unwrap()).add(&layer.biases);
    let activation = if layer.activation == "sigmoid" {
        z.map(activation::sigmoid)
    } else {
        z // linear activation
    };
    activations.push(activation);
}
```

This loop implements the affine transform $z^{(\ell)} = W^{(\ell)} x^{(\ell-1)} + b^{(\ell)}$ followed by the activation $f^{(\ell)}(z^{(\ell)})$.

## Backward Pass

Core excerpt:

```rust
let d_w = delta.mul_transpose(&activations[l]);
let d_b = delta.clone();

let weight_update = d_w.map(|x| -self.learning_rate * x);
let bias_update = d_b.map(|x| -self.learning_rate * x);

self.layers[l].weights = self.layers[l].weights.add(&weight_update);
self.layers[l].biases = self.layers[l].biases.add(&bias_update);
```

- `delta.mul_transpose(&activations[l])` corresponds to $\delta^{(\ell)} (x^{(\ell-1)})^{\top}$.
- `self.layers[l].weights.transpose_mul(&delta)` (not shown above) realises $\left(W^{(\ell)}\right)^{\top}\!\delta^{(\ell)}$ for the next layer's delta.

Bias updates use the fact that biases broadcast over the single-column batch, so the gradient is just `delta` itself.

## Loss Tracking

Every 1 000 epochs the training loop computes mean squared error via repeated calls to `predict`. This mirrors
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$
providing a coarse convergence signal without flooding logs.

## Predict Function

The inference path (`NeuralNetwork::predict`) reuses the same matrix operations as training but skips gradient logic. It converts the final activation back into a flat `Vec<f64>` to make downstream consumption ergonomic.

With the implementation now mapped to equations, the next documents explore the AI-generated `nn-core-library` variant and practical usage workflows.
