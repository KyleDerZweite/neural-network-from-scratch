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

The `nn-core-library` crate provides an optimized implementation using `ndarray` with multiple acceleration options.

### Run with Default Configuration (Rayon Parallelization)

```bash
cargo run --release -p nn-cli -- --task benchmark
```

This uses pure Rust with Rayon for multi-threaded parallelization across CPU cores.

### Run with BLAS Acceleration

For improved performance on larger datasets, enable BLAS support using FlexiBLAS:

```bash
cargo run --release -p nn-cli --features nn-core-library/blas -- --task benchmark
```

**System Requirements:**
- Fedora/RHEL: `sudo dnf install flexiblas-devel`
- Ubuntu/Debian: `sudo apt install libopenblas-dev`
- Arch: `sudo pacman -S openblas`

**Performance Comparison:**

| Task | Without BLAS | With BLAS | Best Choice |
|------|-------------|-----------|-------------|
| XOR (small) | 0.065s | 0.150s | **Pure Rust** (2.3x faster) |
| SIN (large) | 10.226s | 8.283s | **BLAS** (1.23x faster) |

**Key Insight:** BLAS has overhead that makes it slower for small matrices but faster for larger workloads. Use the default (without BLAS) for small networks, and enable BLAS for larger datasets.

### Run Specific Tasks

```bash
# XOR task with nn-core-library
cargo run --release -p nn-cli -- --task xor-library

# SIN approximation with nn-core-library
cargo run --release -p nn-cli -- --task sin-library

# With BLAS enabled
cargo run --release -p nn-cli --features nn-core-library/blas -- --task sin-library
```

### GPU Acceleration (Optional)

Want to try the GPU-enabled version?

```bash
cargo run -p nn-cli --features nn-core-library/gpu -- --task benchmark
```

This enables the `wgpu`-based GPU acceleration. If no compatible GPU is found, the code logs a warning and falls back to CPU execution.

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
