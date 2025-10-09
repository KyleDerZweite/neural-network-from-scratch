//! High-level layer abstraction with optimized initialization.

use crate::activation::Activation;
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, Normal};
use once_cell::sync::Lazy;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Mutex;

static RNG: Lazy<Mutex<SmallRng>> = Lazy::new(|| Mutex::new(SmallRng::seed_from_u64(42)));

#[derive(Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}

impl Layer {
    /// Creates a new layer with proper initialization based on activation function
    pub fn new(input_size: usize, output_size: usize, activation: impl Into<Activation>) -> Self {
        let activation = activation.into();
        let mut rng_guard = RNG.lock().expect("Layer RNG mutex poisoned");

        // Use activation-appropriate initialization
        let scale = activation.init_scale(input_size, output_size);
        let weight_dist = Normal::new(0.0, scale).expect("valid normal distribution");

        // Initialize weights with proper distribution
        let mut weights = Array2::zeros((output_size, input_size));
        for weight in weights.iter_mut() {
            *weight = weight_dist.sample(&mut *rng_guard);
        }

        // Small bias initialization
        let mut biases = Array2::zeros((output_size, 1));
        for bias in biases.iter_mut() {
            *bias = 0.01 * rng_guard.gen::<f64>() - 0.005;
        }

        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass through the layer
    #[inline]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let z = self.weights.dot(input) + &self.biases;
        self.activation.apply(z)
    }

    /// Get number of parameters in this layer
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Gets the weights as a 2D vector.
    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights
            .outer_iter()
            .map(|row| row.to_vec())
            .collect()
    }

    /// Gets the biases as a 1D vector.
    pub fn get_biases(&self) -> Vec<f64> {
        self.biases.iter().copied().collect()
    }

    /// Performs forward pass and returns both activations and weighted sums.
    pub fn forward_with_details(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let input_array = Array2::from_shape_vec((input.len(), 1), input.to_vec())
            .expect("valid input shape");
        let weighted_sum = self.weights.dot(&input_array) + &self.biases;
        let weighted_sum_vec: Vec<f64> = weighted_sum.iter().copied().collect();
        
        let activation_array = self.activation.apply(weighted_sum);
        let activation_vec: Vec<f64> = activation_array.iter().copied().collect();
        
        (activation_vec, weighted_sum_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_dimensions_correct() {
        let layer = Layer::new(10, 5, Activation::ReLU);
        assert_eq!(layer.weights.dim(), (5, 10));
        assert_eq!(layer.biases.dim(), (5, 1));
    }

    #[test]
    fn forward_pass_works() {
        let layer = Layer::new(2, 3, Activation::Linear);
        let input = Array2::ones((2, 1));
        let output = layer.forward(&input);
        assert_eq!(output.dim(), (3, 1));
    }
}
