//! A layer of a neural network.

use crate::matrix::Matrix;
use once_cell::sync::Lazy;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Mutex;

static RNG: Lazy<Mutex<SmallRng>> = Lazy::new(|| Mutex::new(SmallRng::seed_from_u64(42)));

/// A layer of a neural network.
pub struct Layer {
    /// The weights of the layer.
    pub weights: Matrix,
    /// The biases of the layer.
    pub biases: Matrix,
    /// The activation function of the layer.
    pub activation: String,
}

impl Layer {
    /// Creates a new layer.
    ///
    /// # Arguments
    ///
    /// * `input_size` - The number of input neurons.
    /// * `output_size` - The number of output neurons.
    /// * `activation` - The activation function to use.
    pub fn new(input_size: usize, output_size: usize, activation: &str) -> Self {
        let mut rng = RNG.lock().expect("Layer RNG mutex poisoned");
        let init_range = 0.5f64;

        let mut weights_data = Vec::with_capacity(output_size);
        for out_idx in 0..output_size {
            let mut row = Vec::with_capacity(input_size);
            for in_idx in 0..input_size {
                let base = rng.gen_range(-init_range..init_range);
                let offset = ((out_idx * input_size + in_idx) as f64) * 0.001;
                row.push(base + offset);
            }
            weights_data.push(row);
        }

        let mut biases_data = Vec::with_capacity(output_size);
        for out_idx in 0..output_size {
            let base = rng.gen_range(-init_range..init_range);
            let offset = (out_idx as f64) * 0.001;
            biases_data.push(vec![base + offset]);
        }

        Self {
            weights: Matrix::new(output_size, input_size, weights_data),
            biases: Matrix::new(output_size, 1, biases_data),
            activation: activation.to_string(),
        }
    }
}
