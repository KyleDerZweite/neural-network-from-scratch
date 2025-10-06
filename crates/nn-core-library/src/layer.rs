//! High-level layer abstraction built on ndarray.

use crate::activation::Activation;
use ndarray::Array2;
use once_cell::sync::Lazy;
use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::sync::Mutex;

static RNG: Lazy<Mutex<SmallRng>> = Lazy::new(|| Mutex::new(SmallRng::seed_from_u64(1234)));

#[derive(Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: impl Into<Activation>) -> Self {
        let activation = activation.into();
    let mut rng_guard = RNG.lock().expect("Layer RNG mutex poisoned");

        let weight_std = (2.0 / (input_size + output_size) as f64).sqrt();
        let weight_dist = Normal::new(0.0, weight_std).expect("valid normal distribution");
        let bias_dist = Normal::new(0.0, 0.01).expect("valid normal distribution");

        let rng = &mut *rng_guard;
        let mut weights = Array2::zeros((output_size, input_size));
        for weight in weights.iter_mut() {
            *weight = weight_dist.sample(rng);
        }

        let mut biases = Array2::zeros((output_size, 1));
        for bias in biases.iter_mut() {
            *bias = bias_dist.sample(rng);
        }

        Self {
            weights,
            biases,
            activation,
        }
    }
}
