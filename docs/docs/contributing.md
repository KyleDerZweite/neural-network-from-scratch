---
sidebar_position: 7
---

# Contributing

## Development Setup

### Prerequisites
- Rust 1.77 or later
- Cargo
- Git

### Clone and Build
```bash
git clone <repository-url>
cd neuralnet-from-scratch
cargo build
```

### Run Tests
```bash
cargo test
```

### Run Specific Tests
```bash
cargo test --package nn-core
cargo test --package nn-cli
cargo test --package web-interface
```

## Code Organization

### nn-core
- **matrix.rs**: Core matrix operations
- **layer.rs**: Neural network layer implementation
- **network.rs**: Network architecture and training
- **activation.rs**: Activation functions
- **profiling.rs**: Performance utilities
- **lib.rs**: Public API exports

### nn-cli
- **main.rs**: CLI argument parsing and execution

### web-interface
- **main.rs**: Server setup
- **api.rs**: HTTP endpoint handlers
- **models.rs**: Request/response data structures

## Adding New Features

### New Activation Functions
1. Add to `Activation` enum in `activation.rs`
2. Implement activation and derivative functions
3. Update layer forward/backward passes

### New Tasks
1. Add task logic to CLI argument parsing
2. Implement data generation in CLI
3. Add network architecture configuration

### API Endpoints
1. Define request/response models in `models.rs`
2. Implement handler function in `api.rs`
3. Add route to main server setup

## Testing

### Unit Tests
Add tests for individual functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
```

### Integration Tests
Test complete workflows in `tests/` directories.

### Performance Tests
Use the profiling utilities to measure performance:

```rust
#[test]
fn test_training_performance() {
    let mut profiler = Profiler::new();
    // ... setup network ...
    profiler.start("training");
    // ... train ...
    profiler.end("training");
    assert!(profiler.duration("training") < 1000); // Less than 1 second
}
```

## Code Style

### Rust Best Practices
- Use `rustfmt` for formatting
- Follow `clippy` linting suggestions
- Write comprehensive documentation comments
- Use meaningful variable names

### Error Handling
- Use `Result<T, String>` for operations that can fail
- Provide descriptive error messages
- Handle errors gracefully in CLI and API

### Performance
- Minimize allocations in hot paths
- Use efficient data structures
- Profile performance-critical code

## Documentation

### Code Documentation
Use rustdoc comments for public APIs:

```rust
/// Multiplies two matrices using standard matrix multiplication.
/// 
/// # Arguments
/// * `other` - The matrix to multiply with
/// 
/// # Returns
/// Result containing the product matrix or an error message
/// 
/// # Errors
/// Returns an error if matrix dimensions are incompatible
pub fn matmul(&self, other: &Matrix) -> Result<Matrix, String> {
    // implementation
}
```

### User Documentation
- Keep this Docusaurus documentation up to date
- Add examples for new features
- Update API documentation

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Write** tests for new functionality
4. **Implement** the feature
5. **Run** all tests
6. **Update** documentation
7. **Submit** a pull request

## Issue Reporting

When reporting bugs:
- Include Rust version (`rustc --version`)
- Provide steps to reproduce
- Include error messages and stack traces
- Specify operating system

## Future Enhancements

### Potential Features
- Additional activation functions (ReLU, Tanh)
- Different optimizers (Adam, RMSProp)
- Convolutional layers
- Recurrent networks
- Model serialization
- GPU acceleration
- More datasets and tasks

### Architecture Improvements
- Batch training support
- Parallel processing
- Memory optimization
- Advanced profiling