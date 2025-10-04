---
sidebar_position: 6
---

# Examples and Tutorials

## XOR Classification Example

The XOR problem is a classic example that demonstrates why neural networks need hidden layers. Linear classifiers cannot solve XOR, but a neural network with a hidden layer can.

### Problem Description
XOR (exclusive or) returns 1 if exactly one input is 1, and 0 otherwise:

| Input A | Input B | XOR Output |
|---------|---------|------------|
| 0       | 0       | 0          |
| 0       | 1       | 1          |
| 1       | 0       | 1          |
| 1       | 1       | 0          |

### Network Architecture
- **Input Layer**: 2 neurons (for A and B)
- **Hidden Layer**: 2 neurons (with sigmoid activation)
- **Output Layer**: 1 neuron (sigmoid activation)

### Training
```bash
cargo run --bin nn-cli -- --task xor --epochs 10000
```

### Expected Output
After training, the network should produce outputs close to the target values:
- (0,0) → ~0.0
- (0,1) → ~1.0
- (1,0) → ~1.0
- (1,1) → ~0.0

## Sine Function Approximation

This example demonstrates function approximation using neural networks.

### Problem Description
Approximate the sine function over the interval [0, 7] using a neural network.

### Network Architecture
- **Input Layer**: 1 neuron (x value)
- **Hidden Layer**: 32 neurons (sigmoid activation)
- **Output Layer**: 1 neuron (linear activation)

### Training Data
- 200 points uniformly sampled from [0, 7]
- Target values: sin(x) for each x

### Training
```bash
cargo run --bin nn-cli -- --task sin --epochs 20000 --learning-rate 0.01
```

### Visualization
After training, a plot is generated showing:
- Training data points
- Network predictions
- Training loss curve

## Custom Network Creation

### Using the Library Directly
```rust
use nn_core::{Network, Layer, Matrix, Activation};

// Create a simple 2-2-1 network
let mut network = Network::new(vec![
    Layer::new(2, 2, Activation::Sigmoid),
    Layer::new(2, 1, Activation::Sigmoid),
], 0.1);

// Training data for XOR
let inputs = Matrix::from_vec(4, 2, vec![
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
]);
let targets = Matrix::from_vec(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

// Train the network
let losses = network.train(&inputs, &targets, 10000)?;
```

### Custom Activation Functions
Currently only sigmoid is implemented, but the architecture supports extending with new activations:

```rust
pub enum Activation {
    Sigmoid,
    // Future: ReLU, Tanh, etc.
}
```

## Hyperparameter Tuning

### Learning Rate
- **Too high**: Training may diverge or oscillate
- **Too low**: Training is slow
- **Good range**: 0.001 to 0.1

### Number of Epochs
- Depends on problem complexity
- XOR: 10,000 epochs typically sufficient
- Sine: 20,000+ epochs for good approximation

### Hidden Layer Size
- **Too small**: Underfitting (cannot learn complex patterns)
- **Too large**: Overfitting (memorizes training data)
- **XOR**: 2-3 neurons sufficient
- **Sine**: 32+ neurons for smooth approximation

## Troubleshooting

### Training Doesn't Converge
1. **Check learning rate**: Try reducing by factor of 10
2. **Increase epochs**: Allow more training time
3. **Verify data**: Ensure inputs/targets are correct
4. **Check network size**: May need more hidden neurons

### Poor Performance
1. **Overfitting**: Reduce hidden layer size or add regularization
2. **Underfitting**: Increase hidden layer size
3. **Data issues**: Check for outliers or incorrect labels

### Memory Issues
- Large networks or datasets may cause high memory usage
- Consider batch training for very large datasets
- Monitor with the profiling utilities

## Advanced Usage

### Profiling Performance
```rust
use nn_core::profiling::Profiler;

let mut profiler = Profiler::new();
profiler.start("training");

// ... training code ...

profiler.end("training");
println!("Training time: {} ms", profiler.duration("training"));
```

### Custom Loss Functions
The current implementation uses Mean Squared Error. To add custom losses:

```rust
pub fn custom_loss(predictions: &Matrix, targets: &Matrix) -> f64 {
    // Implement custom loss calculation
    // Return scalar loss value
}
```

### Saving/Loading Networks
Future enhancements may include serialization for saving trained networks to disk.