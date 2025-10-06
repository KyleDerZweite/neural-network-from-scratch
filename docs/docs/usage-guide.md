---
title: Usage and Experimentation Guide
sidebar_position: 6
description: Step-by-step instructions for running and extending nn-core.
---

# Usage and Experimentation Guide

This guide explains how to build, run, and extend the project locally. Commands assume a recent Rust toolchain (`rustup` >= 1.76) and Node.js ≥ 20 for the documentation workspace.

## 1. Clone and Build

```bash
# from your workspace root
cargo build
```

## 2. Run the CLI Demo

The `nn-cli` crate wraps `nn-core` with a simple command-line interface.

```bash
cargo run -p nn-cli -- --help
```

Typical XOR training run:

```bash
cargo run -p nn-cli -- train xor --epochs 5000 --learning-rate 0.5
```

## 3. Profile Hotspots (Optional)

Enable the `profiling` feature to record scoped timings during training:

```bash
cargo run -p nn-cli --features profiling -- train xor
```

Timing information prints to stdout, highlighting the cost of matrix operations versus activation functions.

## 4. Explore the AI-Assisted Variant

Want to try the GPU-enabled version?

```bash
cargo run -p nn-cli --no-default-features --features gpu -- train xor
```

This toggles the `nn-core-library` backend with optional GPU acceleration. If no compatible GPU is found, the code logs a warning and falls back to CPU execution.

## 5. Run Tests

```bash
cargo test
```

This executes unit tests across all crates, verifying matrix identities, layer initialisation, and end-to-end training convergence.

## 6. Generate Documentation Site

The Docusaurus site lives in the `docs/` directory. To work on the site locally:

```bash
cd docs
npm install
npm run start
```

Visit `http://localhost:3000` to browse the rendered documentation. Changes in the `docs/docs/` folder reload automatically.

## 7. Extend the Project

Ideas for further experiments:

- Add ReLU or tanh activations by implementing new derivatives in `activation.rs`.
- Introduce mini-batching by extending the `Matrix` operators to handle multiple columns.
- Port the network to 32-bit floats to compare convergence and performance trade-offs.

Refer back to the [Mathematical Foundations](./math-foundations) and [Implementation Walkthrough](./implementation-details) sections for guidance while making modifications.
