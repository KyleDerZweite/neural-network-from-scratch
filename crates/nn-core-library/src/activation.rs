//! Activation helpers wrapping ndarray primitives.

use ndarray::Array2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Sigmoid,
}

impl Activation {
    pub fn from_str(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "sigmoid" => Activation::Sigmoid,
            _ => Activation::Linear,
        }
    }

    pub fn apply(&self, input: Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Linear => input,
            Activation::Sigmoid => sigmoid_matrix(&input),
        }
    }

    pub fn derivative_from_output(&self, output: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Linear => Array2::ones(output.raw_dim()),
            Activation::Sigmoid => sigmoid_derivative_from_output(output),
        }
    }
}

impl From<&str> for Activation {
    fn from(value: &str) -> Self {
        Activation::from_str(value)
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    matrix.mapv(sigmoid)
}

pub fn sigmoid_derivative_from_output(output: &Array2<f64>) -> Array2<f64> {
    output.mapv(|y| y * (1.0 - y))
}
