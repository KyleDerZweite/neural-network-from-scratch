---
sidebar_position: 3
---

# Command Line Interface

The `nn-cli` crate provides a simple command-line interface for training neural networks on predefined tasks.

## Usage

```bash
cargo run --bin nn-cli -- [OPTIONS]
```

## Options

- `--task <TASK>`: Specify the training task
  - `xor`: Train on XOR classification problem
  - `sin`: Train on sine function approximation
- `--learning-rate <RATE>`: Learning rate for gradient descent (default: 0.1)
- `--epochs <NUM>`: Number of training epochs (default: 10000)
- `--hidden <NUM>`: Number of neurons in hidden layer (default: 32 for sin, 2 for xor)
- `--help`: Display help information

## Examples

### XOR Classification
Train a network to solve the XOR problem with default settings:

```bash
cargo run --bin nn-cli -- --task xor
```

Custom learning rate and epochs:

```bash
cargo run --bin nn-cli -- --task xor --learning-rate 0.01 --epochs 50000
```

### Sine Approximation
Train a network to approximate the sine function:

```bash
cargo run --bin nn-cli -- --task sin
```

With custom hidden layer size:

```bash
cargo run --bin nn-cli -- --task sin --hidden 64 --epochs 20000
```

## Output

During training, the CLI displays:
- Current epoch number
- Loss value at regular intervals
- Final training results

After training completes:
- A plot of training loss over epochs (saved as PNG)
- Final network predictions on test data

## Error Handling

The CLI provides user-friendly error messages for:
- Invalid task names
- Invalid hyperparameter values (clamped to valid ranges)
- Training failures

## Tasks

### XOR Classification
- **Input**: 2 binary values (0 or 1)
- **Output**: XOR result (0 or 1)
- **Network**: 2-2-1 architecture
- **Training Data**: All 4 possible input combinations
- **Goal**: Learn the XOR logic gate

### Sine Approximation
- **Input**: Single value x in range [0, 7]
- **Output**: sin(x)
- **Network**: 1-32-1 architecture
- **Training Data**: 200 uniformly sampled points
- **Goal**: Approximate the sine function

## Implementation Details

The CLI:
1. Parses command-line arguments
2. Initializes the appropriate network architecture
3. Generates training data
4. Runs the training loop with progress reporting
5. Saves results and generates plots