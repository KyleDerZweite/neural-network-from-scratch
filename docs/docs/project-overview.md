---
title: Project Overview
sidebar_position: 1
description: High-level tour of the nn-core project goals and structure.
---

# Project Overview

Welcome to the research hub for the `nn-core` crate. This Rust project builds a feed-forward neural network *from first principles* in order to deepen intuition about optimisation, linear algebra, and numerical stability. The implementation deliberately avoids heavyweight dependencies so that every moving part remains inspectable.

## Goals

- **Pedagogical clarity**: expose the exact computations involved in forward passes, loss evaluation, and gradient flow.
- **Determinism**: use a reproducible pseudo-random generator so experiments can be repeated bit-for-bit.
- **Extensibility**: provide clean abstractions (matrix, layer, network) that can be instrumented, profiled, or replaced.

## Crate Layout

- [`crates/nn-core`](https://github.com/KyleDerZweite/neural-network-from-scratch/tree/master/crates/nn-core) — the primary, hand-crafted implementation analysed throughout this documentation set.
- [`crates/nn-core-library`](https://github.com/KyleDerZweite/neural-network-from-scratch/tree/master/crates/nn-core-library) — an AI-generated sibling crate exploring additional features (GPU acceleration, advanced optimisers). This variant is discussed separately in [AI-Assisted Variant](./nn-core-library).
- [`crates/nn-cli`](https://github.com/KyleDerZweite/neural-network-from-scratch/tree/master/crates/nn-cli) — a command-line interface for training and evaluating networks built atop `nn-core`.

## Reading Guide

This research documentation is organised into the following modules:

1. **Approach and Design Principles** — why the architecture looks the way it does.
2. **Mathematical Foundations** — the linear algebra and calculus underpinning the code.
3. **Implementation Walkthrough** — how each formula maps to Rust.
4. **AI-Assisted Variant** — what changes in the `nn-core-library` experiment.
5. **Usage Guide** — how to run, train, and extend the project locally.

Feel free to dip into any section that matches your current investigation. The sections use matching notation so you can translate seamlessly between docs and source.
