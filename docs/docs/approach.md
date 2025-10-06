---
title: Approach and Design Principles
sidebar_position: 2
description: Architectural mindset behind the nn-core neural network.
---

# Approach and Design Principles

The `nn-core` crate embraces a "glass box" philosophy: every transformation in the network is explicit, type-checked, and easy to instrument. This section outlines the guiding principles behind the architecture and how they inform key engineering decisions.

## Minimal Surface Area

The codebase is intentionally compact. Rather than rely on high-level tensor libraries, `nn-core` implements just enough linear algebra to express dense layers:

- A custom [`Matrix`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/matrix.rs) type for numerical data.
- Deterministic [`Layer`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/layer.rs) initialisation for replicable experiments.
- A [`NeuralNetwork`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core/src/network.rs) orchestration layer that stitches components together.

This minimal surface area keeps cognitive load low while still capturing the essential mechanics of gradient-based learning.

## Deterministic Experiments

Reproducibility is critical when experimenting with learning rates, activation functions, or weight initialisation strategies. `nn-core` seeds a `SmallRng` via `once_cell::sync::Lazy`, guaranteeing that the same weight matrices and biases are generated on every run unless you change the seed.

## Profiling from the Start

Every numerically heavy operation is wrapped in the `profile_scope!` macro. Enabling the `profiling` feature toggles scoped timing, making it trivial to spot hotspots when you experiment with larger networks or batch sizes.

## Clear Separation of Concerns

- **Matrix layer**: handles raw linear algebra.
- **Activation layer**: holds nonlinearities and their derivatives.
- **Network layer**: orchestrates forward/backward passes and gradient descent.

Each layer exposes a narrow API, allowing you to swap implementations (for example, replacing the matrix backend with BLAS bindings) without rewriting the learning logic.

## Research-Friendly Defaults

- **Learning Rate**: treated as a global scalar for simplicity. More sophisticated schedules can be layered on later.
- **Activation Functions**: sigmoid and linear keep the mathematics approachable; you can add ReLU or tanh alongside derivative rules as extensions.
- **Loss Reporting**: prints mean squared error every 1 000 epochs by default to reduce console noise while still signalling convergence trends.

Together these choices make `nn-core` ideal for coursework, scratchpad experimentation, and teaching demonstrations.
