//! Layers leveraging the optimised matrix backend.

use crate::matrix::Matrix;
use once_cell::sync::Lazy;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Mutex;

static RNG: Lazy<Mutex<SmallRng>> = Lazy::new(|| Mutex::new(SmallRng::seed_from_u64(42)));

/// A fully-connected layer.
#[derive(Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: ActivationKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    Linear,
    Sigmoid,
}

impl From<&str> for ActivationKind {
    fn from(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "sigmoid" => ActivationKind::Sigmoid,
            _ => ActivationKind::Linear,
        }
    }
}

impl Layer {
    /// Creates a new layer with deterministic initialisation.
    pub fn new(input_size: usize, output_size: usize, activation: impl Into<ActivationKind>) -> Self {
        let activation = activation.into();
        let mut rng = RNG.lock().expect("Layer RNG mutex poisoned");
        let init_range = 0.5f64;

        let mut weights = Vec::with_capacity(output_size * input_size);
        for out_idx in 0..output_size {
            for in_idx in 0..input_size {
                let base = rng.gen_range(-init_range..init_range);
                let offset = ((out_idx * input_size + in_idx) as f64) * 0.001;
                weights.push(base + offset);
            }
        }

        let mut biases = Vec::with_capacity(output_size);
        for out_idx in 0..output_size {
            let base = rng.gen_range(-init_range..init_range);
            let offset = (out_idx as f64) * 0.001;
            biases.push(base + offset);
        }

        Self {
            weights: Matrix::new(output_size, input_size, weights),
            biases: Matrix::new(output_size, 1, biases),
            activation,
        }
    }
}
