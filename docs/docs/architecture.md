---
sidebar_position: 2
---

# Architecture Overview

## Project Structure

The Neural Network from Scratch project is organized as a Cargo workspace with three main crates:

### nn-core
The core neural network library containing all the fundamental implementations:

- **matrix.rs**: Custom matrix data structure with operations like multiplication, addition, transposition
- **layer.rs**: Neural network layer implementation with weights, biases, and forward/backward passes
- **network.rs**: Complete neural network architecture with training capabilities
- **activation.rs**: Activation functions (currently sigmoid)
- **profiling.rs**: Performance measurement utilities

### nn-cli
Command-line interface for running training tasks:

- Simple argument parsing for task selection and hyperparameters
- Integration with nn-core for training execution
- Output formatting and progress reporting

### web-interface
RESTful web API and static file serving:

- **api.rs**: HTTP endpoints for network operations
- **models.rs**: Data structures for API requests/responses
- **main.rs**: Server setup and routing

## Data Flow

### Training Process
1. **Input Processing**: CLI parses arguments or API receives request
2. **Network Initialization**: Create network with specified topology
3. **Data Preparation**: Generate or load training data
4. **Training Loop**:
   - Forward pass through layers
   - Calculate loss
   - Backward pass (backpropagation)
   - Update weights and biases
5. **Output**: Training metrics, plots, and final network state

### Key Components

#### Matrix Operations
All linear algebra is implemented from scratch:

```rust
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec&lt;f64&gt;,
}
```

Operations include:
- Matrix multiplication (`matmul`)
- Element-wise operations
- Transposition
- Random initialization

#### Layer Implementation
Each layer handles one step of the forward and backward passes:

```rust
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Activation,
}
```

#### Network Architecture
The `Network` struct manages the collection of layers and coordinates training:

```rust
pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}
```

## Design Principles

### From Scratch Philosophy
- No external ML libraries (no ndarray, no tch, etc.)
- Custom implementations of all mathematical operations
- Educational focus on understanding fundamentals

### Performance Considerations
- Efficient matrix operations using flat `Vec<f64>`
- Minimal allocations during training loops
- Profiling utilities for performance monitoring

### Modularity
- Clear separation between core logic, CLI, and web interface
- Reusable components across different interfaces
- Testable units with comprehensive integration tests