---
sidebar_position: 4
---

# Core Implementation

This section provides a detailed explanation of the neural network implementation in the `nn-core` crate.

## Matrix Operations

### Data Structure
```rust
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec&lt;f64&gt;,
}
```

The `Matrix` struct uses a flat `Vec<f64>` for efficient memory layout and cache performance.

### Key Operations

#### Matrix Multiplication
```rust
impl Matrix {
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err("Matrix dimension mismatch".to_string());
        }
        
        let mut result = Matrix::zeros(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i * result.cols + j] += 
                        self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
            }
        }
        
        Ok(result)
    }
}
```

#### Element-wise Operations
```rust
pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
    if self.rows != other.rows || self.cols != other.cols {
        return Err("Matrix dimension mismatch".to_string());
    }
    
    let mut result = Matrix::zeros(self.rows, self.cols);
    for i in 0..self.data.len() {
        result.data[i] = self.data[i] + other.data[i];
    }
    Ok(result)
}
```

#### Transposition
```rust
pub fn transpose(&self) -> Matrix {
    let mut result = Matrix::zeros(self.cols, self.rows);
    for i in 0..self.rows {
        for j in 0..self.cols {
            result.data[j * result.rows + i] = self.data[i * self.cols + j];
        }
    }
    result
}
```

## Activation Functions

### Sigmoid Implementation
```rust
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}
```

The sigmoid function maps any real value to the range (0, 1), making it suitable for binary classification.

## Layer Implementation

### Structure
```rust
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Activation,
}
```

Each layer contains:
- **Weights**: Matrix connecting inputs to outputs
- **Biases**: Vector added to weighted inputs
- **Activation**: Function applied element-wise to outputs

### Forward Pass
```rust
pub fn forward(&self, input: &Matrix) -> Result<Matrix, String> {
    // W * x + b
    let weighted = self.weights.matmul(input)?;
    let biased = weighted.add(&self.biases)?;
    
    // Apply activation
    let activated = biased.apply_activation(self.activation);
    
    Ok(activated)
}
```

### Backward Pass
```rust
pub fn backward(&mut self, input: &Matrix, output_error: &Matrix, learning_rate: f64) -> Result<Matrix, String> {
    // Compute activation derivative
    let activated = self.forward(input)?;
    let activation_derivative = activated.activation_derivative(self.activation);
    
    // Element-wise multiply with output error
    let delta = output_error.elementwise_multiply(&activation_derivative)?;
    
    // Compute weight and bias gradients
    let input_transposed = input.transpose();
    let weight_gradients = delta.matmul(&input_transposed)?;
    
    // Update weights and biases
    let weight_updates = weight_gradients.scalar_multiply(learning_rate);
    self.weights = self.weights.subtract(&weight_updates)?;
    
    let bias_updates = delta.mean_axis(1)?.scalar_multiply(learning_rate);
    self.biases = self.biases.subtract(&bias_updates)?;
    
    // Compute input error for previous layer
    let weights_transposed = self.weights.transpose();
    let input_error = weights_transposed.matmul(&delta)?;
    
    Ok(input_error)
}
```

## Network Architecture

### Structure
```rust
pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}
```

### Training Process

#### Forward Pass
```rust
pub fn forward(&self, input: &Matrix) -> Result<Matrix, String> {
    let mut current = input.clone();
    for layer in &self.layers {
        current = layer.forward(&current)?;
    }
    Ok(current)
}
```

#### Backward Pass (Backpropagation)
```rust
pub fn backward(&mut self, input: &Matrix, target: &Matrix) -> Result<f64, String> {
    // Forward pass
    let output = self.forward(input)?;
    
    // Compute loss (Mean Squared Error)
    let error = output.subtract(target)?;
    let loss = error.elementwise_multiply(&error)?.mean()?;
    
    // Backward pass
    let mut current_error = error;
    for i in (0..self.layers.len()).rev() {
        current_error = self.layers[i].backward(input, &current_error, self.learning_rate)?;
    }
    
    Ok(loss)
}
```

#### Training Loop
```rust
pub fn train(&mut self, inputs: &Matrix, targets: &Matrix, epochs: usize) -> Result<Vec<f64>, String> {
    let mut losses = Vec::new();
    
    for epoch in 0..epochs {
        let loss = self.backward(inputs, targets)?;
        losses.push(loss);
        
        if epoch % 1000 == 0 {
            println!("Epoch {}, Loss: {:.6}", epoch, loss);
        }
    }
    
    Ok(losses)
}
```

## Mathematical Foundations

### Feed-Forward Neural Networks
A feed-forward neural network consists of layers of neurons where connections only flow forward. Each neuron computes a weighted sum of its inputs plus a bias, then applies an activation function.

### Backpropagation
Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to the network parameters.

#### Chain Rule
For a network with parameters θ, loss L:
```
∂L/∂θ = ∂L/∂y * ∂y/∂θ
```

Where y is the output of a layer, and θ are the weights/biases.

### Gradient Descent
Parameters are updated using the gradient:
```
θ_new = θ_old - learning_rate * ∂L/∂θ
```

## Performance Optimizations

- **Flat Vector Storage**: Reduces memory allocations and improves cache locality
- **In-Place Operations**: Minimize memory usage during computation
- **Efficient Matrix Multiplication**: Optimized triple-loop implementation
- **Profiling Utilities**: Built-in timing and performance measurement