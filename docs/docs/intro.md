---
sidebar_position: 1
---

# Neural Network from Scratch

Welcome to **Neural Network from Scratch**, a comprehensive implementation of neural networks built entirely from scratch in Rust. This project demonstrates the fundamental principles of machine learning by implementing feed-forward neural networks without relying on any external machine learning libraries.

## What is this project?

This project provides:

- **Core Neural Network Library** (`nn-core`): A complete implementation of feed-forward neural networks with backpropagation
- **Command Line Interface** (`nn-cli`): Easy-to-use CLI for training and testing networks on common problems
- **Web Interface** (`web-interface`): RESTful API and web UI for interactive neural network experimentation

## Key Features

- ✅ **From Scratch Implementation**: All neural network components implemented without external ML libraries
- ✅ **Custom Linear Algebra**: Matrix and vector operations built from scratch in Rust
- ✅ **Backpropagation**: Full gradient descent training algorithm
- ✅ **Multiple Tasks**: XOR classification and sine function approximation
- ✅ **Configurable Architecture**: Adjustable network topology, learning rate, and epochs
- ✅ **Visualization**: Training loss plots and performance metrics
- ✅ **Web API**: RESTful endpoints for programmatic access

## Quick Start

### Prerequisites
- Rust 1.77 or later
- Cargo package manager

### Running XOR Classification
```bash
cargo run --bin nn-cli -- --task xor
```

### Running Sine Approximation
```bash
cargo run --bin nn-cli -- --task sin
```

### Starting the Web Interface
```bash
cargo run --bin web-interface
```

## Project Structure

```
├── nn-core/          # Core neural network library
│   ├── src/
│   │   ├── lib.rs
│   │   ├── matrix.rs     # Custom matrix implementation
│   │   ├── layer.rs      # Neural network layers
│   │   ├── network.rs    # Network architecture
│   │   ├── activation.rs # Activation functions
│   │   └── profiling.rs  # Performance profiling
│   └── tests/
├── nn-cli/           # Command line interface
│   └── src/main.rs
├── web-interface/    # Web API and UI
│   ├── src/
│   │   ├── main.rs
│   │   ├── api.rs
│   │   └── models.rs
│   └── static/       # Web assets
└── docs/             # This documentation
```

## Learning Objectives

This project serves as an educational tool for understanding:

- How neural networks work at the mathematical level
- Implementation of backpropagation algorithm
- Matrix operations for efficient computation
- Rust programming for high-performance computing
- Building maintainable and testable code architectures
