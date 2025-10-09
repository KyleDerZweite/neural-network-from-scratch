# A From-Scratch Implementation of a Feed-Forward Neural Network for Classification and Function Approximation

## 1.0 Project Overview

### 1.1 Introduction and Motivation
This document provides a comprehensive overview of a feed-forward neural network implemented entirely from scratch. This project serves as the practical component for the "Neuroinformatik" module at Hochschule Ruhr West (HRW), developed under the academic guidance of Prof. Dr.-Ing. Uwe Handmann.

The central tenet of this project is to build a functional neural network from fundamental principles. In strict adherence to the assignment's requirements, the use of external machine learning libraries and frameworks (such as TensorFlow, PyTorch, or scikit-learn) has been explicitly forbidden. All core components—from the network's data structures to the backpropagation learning algorithm and even fundamental linear algebra operations like matrix multiplication—have been programmed manually. This "from-scratch" approach is a pedagogical imperative designed to deconstruct the "black box" nature of neural networks and foster a deep, foundational understanding of their internal mechanics. This philosophy aligns with the educational goal of not just using intelligent systems, but truly comprehending their construction and operation.

Given the course's emphasis on a detailed explanation of the program code during the final oral examination, this README is structured as more than a simple user manual. It functions as a technical design document and a theoretical exposition, systematically linking the implemented code components back to the core principles of neuroinformatics discussed in the lectures.

### 1.2 Project Scope and Objectives
The scope of this project encompasses the design, implementation, training, and evaluation of a Multi-Layer Perceptron (MLP). The primary objectives are:

- Implementation: Develop a configurable feed-forward neural network with a single hidden layer in a high-level programming language.
- Training: Implement the backpropagation algorithm as the mechanism for supervised learning, utilizing gradient descent to iteratively optimize the network's weights and biases.
- Application & Demonstration: Train and validate the network on two classic benchmark problems:
  - Logical Classification: Solving the non-linearly separable Exclusive OR (XOR) problem, defined as `y = XOR(x1, x2)`.
  - Function Approximation: Approximating the continuous sine function, `y = sin(x)`, over a defined interval (unspecified).

The selection of these two problems is deliberate. It is designed to demonstrate the network's versatility and its capacity as a universal approximator. By successfully tackling both a discrete classification task and a continuous regression task, the implementation validates its ability to handle the two fundamental problem classes in supervised machine learning.

## 2.0 Theoretical Foundations

### 2.1 The Perceptron and its Limitations: The XOR Dilemma
The architecture of this project is rooted in the historical development of artificial neural networks, which began with biologically inspired models like the McCulloch-Pitts neuron and Frank Rosenblatt's Perceptron. A simple Perceptron computes a weighted sum of its inputs and passes the result through a step function. Geometrically, this means a single perceptron, or a single layer of perceptrons, can only function as a linear classifier. It learns to define a single hyperplane (a line in two dimensions) to separate the input space into two categories.

This inherent linearity imposes a critical limitation: single-layer perceptrons can only solve problems that are linearly separable. The Exclusive OR (XOR) function is the canonical example of a problem that fails this criterion. As illustrated in the course materials, no single straight line can be drawn in the 2D input space to correctly separate the points where the output is 1 (`(0,1)` and `(1,0)`) from the points where the output is 0 (`(0,0)` and `(1,1)`). This fundamental inability, famously highlighted in the 1969 book Perceptrons by Minsky and Papert, became known as the "XOR Dilemma" and demonstrated the necessity of more complex network architectures.

### 2.2 The Multi-Layer Perceptron (MLP) Architecture
The solution to the limitations of the simple perceptron is the Multi-Layer Perceptron (MLP). An MLP is a class of feed-forward neural network characterized by the inclusion of one or more hidden layers positioned between the input and output layers. This project implements an MLP with the specified three-layer topology:

- Input Layer: Receives the raw input vector. Size corresponds to the dimensionality of the input data (e.g., two neurons for the XOR problem, one for the `sin(x)` problem).
- Hidden Layer: Grants the network its ability to learn non-linear relationships. Each hidden neuron can be thought of as defining its own linear boundary. Combining these in the subsequent layer allows complex, non-linear decision regions, thereby overcoming the XOR dilemma.
- Output Layer: Produces the final prediction (e.g., one neuron for both the XOR and `sin(x)` problems).

In a feed-forward network, the signal propagates unidirectionally from input to output. The output `s_j^k` of a neuron `j` in layer `k` is:
```
s_j^k = φ(v_j^k) = φ(Σ_{i=0..I} w_{ji} * s_i^{(k−1)})
```
where `w_{ji}` are the weights connecting neuron `i` in layer `(k−1)` to neuron `j` in layer `k`, and `i=0` represents the bias term.

### 2.3 The Backpropagation Learning Algorithm
Training an MLP involves finding weights and biases that minimize the error between predictions and targets. This project employs Backpropagation (gradient descent). For each training example:

1. Forward Pass: Present input and compute outputs layer-by-layer.
2. Error Calculation: Compare output to target via a loss (e.g., Mean Squared Error).
3. Backward Pass: Propagate error backward using the chain rule to compute gradients `∂E/∂w_ji`.
4. Weight Update: Update parameters opposite the gradient:
```
w_ji(new) = w_ji(old) − η * ∂E/∂w_ji
```
where `η` is the learning rate.

#### Important aspects (summary)
- Gradient computation uses the chain rule layer-by-layer; cost is on the same order as a forward pass.
- Loss functions commonly used: mean squared error (regression) and cross-entropy (classification).
- Training modes: batch, stochastic (per sample), and mini-batch (typical in practice).
- Optimizer enhancements: momentum and adaptive learning rates (e.g., RMSProp/Adam concepts), though plain gradient descent is implemented here.
- Initialization must break symmetry; small random weights are standard.
- Activation choice affects gradient flow; sigmoid/tanh can cause vanishing gradients in deeper nets; ReLU-family mitigates this (project uses sigmoid as specified).
- Regularization options include L2 weight decay and early stopping to reduce overfitting (not required by this project).
- Stopping criteria: max epochs, validation loss plateau, or small gradient norm.

Source: https://de.wikipedia.org/wiki/Backpropagation

## 3.0 Implementation Architecture and Details

### 3.1 Core Data Structures: Representing the Network
The network is encapsulated in a `NeuralNetwork` class that manages layers, hyperparameters, and training. The network is composed of `Layer` objects, each storing its weight matrix, bias vector, and activation function.

Weights connecting a layer with `I` neurons to a subsequent layer with `J` neurons are represented as a `J×I` matrix. Biases for the `J` neurons are stored in a `J×1` vector. All weights are initialized with small random values to break symmetry.

### 3.2 The "From-Scratch" Linear Algebra Engine
A dedicated module implements all required matrix/vector operations:

- Matrix multiplication (forward pass `v = W·s` and backward propagation).
- Matrix-vector addition (adding bias `v = W·s + b`).
- Element-wise (Hadamard) product.
- Matrix transposition.
- Element-wise application of functions (activation and derivatives).

### 3.3 Activation Functions and Their Derivatives
Non-linear activations are essential. This implementation includes the sigmoid function:

- Sigmoid: `φ(v) = 1 / (1 + e^(−v))`
- Derivative: `φ'(v) = φ(v) * (1 − φ(v))`

### 3.4 The Backpropagation Algorithm in Code
The `train` method orchestrates learning over epochs. For each example:

- Forward Pass: For each layer, compute `output = activation(W * input + b)`. Store activations and pre-activations.
- Backward Pass:
  - Compute output layer delta: `(predicted − target) ⊙ activation'(pre-activation)`.
  - For each hidden layer (last to first): `delta = (W_next^T * delta_next) ⊙ activation'(pre-activation_hidden)`.
  - Compute gradients: `∂E/∂W = delta * activation_prev^T`, `∂E/∂b = delta`.
  - Update `W` and `b` using the learning rate.

## 4.0 Application I: Solving the XOR Classification Problem

### 4.1 Problem Formulation and Network Topology
- Task: Train the network to map XOR input patterns to binary outputs.
- Data: `(0,0) → 0`, `(0,1) → 1`, `(1,0) → 1`, `(1,1) → 0`.
- Network: `2-2-1` topology
  - Input: 2 neurons (`x1`, `x2`)
  - Hidden: 2 neurons
  - Output: 1 neuron with sigmoid

### 4.2 Training Hyperparameters
| Hyperparameter                | Value    |
|------------------------------|----------|
| Network Topology             | 2-2-1    |
| Learning Rate (η)            | 0.1      |
| Number of Epochs             | 10,000   |
| Activation Function (Hidden) | Sigmoid  |
| Activation Function (Output) | Sigmoid  |
| Loss Function                | MSE      |

### 4.3 Results and Verification
After training, predictions for the four XOR patterns:

| Input (x1, x2) | Target | Predicted (Raw) | Predicted (Classified) |
|----------------|--------|------------------|------------------------|
| (0, 0)         | 0      | 0.051            | 0                      |
| (0, 1)         | 1      | 0.953            | 1                      |
| (1, 0)         | 1      | 0.953            | 1                      |
| (1, 1)         | 0      | 0.056            | 0                      |

## 5.0 Application II: Approximating the Sine Function

### 5.1 Problem Formulation and Network Topology
- Task: Approximate `y = sin(x)` over the continuous interval (unspecified).
- Network: `1-32-1` topology
  - Input: 1 neuron (`x`)
  - Hidden: 32 neurons
  - Output: 1 neuron with linear activation (no activation)

### 5.2 Data Generation and Normalization
- Training Data: 200 points sampled uniformly from an interval (unspecified). Targets computed as `sin(x)`.
- Normalization: Inputs normalized to a standard range (exact range unspecified). Targets in `[-1, 1]` not scaled. The normalization function was stored for consistent prediction and plotting.

### 5.3 Training and Evaluation
Trained using backpropagation with MSE. Learning rate `0.01`, `20,000` epochs. MSE monitored for convergence.

## 6.0 Usage and Execution Guide

### 6.1 Prerequisites
- Programming Language: Rust 1.76+ (install via [rustup](https://rustup.rs/))
- Visualization: `plotters` crate for generating training plots (included as dependency)
- Optional BLAS acceleration: `flexiblas-devel` package for enhanced matrix operations
  - Fedora/RHEL: `sudo dnf install flexiblas-devel`
  - Ubuntu/Debian: `sudo apt install libopenblas-dev`
  - Arch: `sudo pacman -S openblas`

All core components are implemented in Rust with no external machine learning libraries.

### 6.2 Code Structure
- `crates/nn-cli/src/main.rs`: CLI interface for training and benchmarking neural networks
- `crates/nn-core/src/`: Basic neural network implementation with custom matrix operations
- `crates/nn-core-library/src/`: Optimized implementation using `ndarray` with multiple acceleration options
- `crates/nn-core-library/src/network.rs`: Neural network training logic with backpropagation
- `crates/nn-core-library/src/layer.rs`: Layer abstraction with weight initialization
- `crates/nn-core-library/src/activation.rs`: Activation functions and derivatives
- `crates/nn-core-library/src/optimizer.rs`: Gradient descent optimizers (SGD, Adam, RMSprop)

### 6.3 Running the Program
To train and test the network on the XOR problem:
```bash
cargo run --release -p nn-cli -- --task xor-library
```

To train on the `sin(x)` approximation problem and generate the plot:
```bash
cargo run --release -p nn-cli -- --task sin-library
```

To run a full benchmark comparing all implementations:
```bash
cargo run --release -p nn-cli -- --task benchmark
```

To enable BLAS acceleration for improved performance on larger datasets:
```bash
cargo run --release -p nn-cli --features nn-core-library/blas -- --task benchmark
```

## 7.0 Conclusion
This project successfully created a fully functional feed-forward neural network from scratch. Backpropagation and all underlying matrix operations were implemented without external ML libraries, providing insight into the fundamental mechanics of neural networks.

The network solved both a canonical non-linear classification problem (XOR) and a continuous function approximation problem (`sin(x)`), demonstrating versatility consistent with the universal approximation theorem. Building this system from first principles reinforced a practical understanding of gradient-based learning and the interplay between architecture, activation functions, and backpropagation.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
