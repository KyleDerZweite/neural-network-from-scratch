//! Activation functions and their derivatives.

/// The sigmoid activation function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The derivative of the sigmoid activation function.
pub fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

/// The derivative of the sigmoid activation function, calculated from the output of the sigmoid function.
pub fn sigmoid_derivative_from_output(output: f64) -> f64 {
    output * (1.0 - output)
}
