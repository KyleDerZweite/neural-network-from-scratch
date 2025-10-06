//! Activation functions with vectorized implementations.

use ndarray::Array2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Linear,
}

impl Activation {
    /// Apply activation function to array (vectorized)
    pub fn apply(&self, mut z: Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => {
                z.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                z
            }
            Activation::Tanh => {
                z.mapv_inplace(|x| x.tanh());
                z
            }
            Activation::ReLU => {
                z.mapv_inplace(|x| x.max(0.0));
                z
            }
            Activation::LeakyReLU => {
                z.mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x });
                z
            }
            Activation::Linear => z,
        }
    }

    /// Compute derivative from activation output (vectorized)
    pub fn derivative(&self, activated: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => activated * &(1.0 - activated),
            Activation::Tanh => 1.0 - activated.mapv(|x| x * x),
            Activation::ReLU => activated.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU => activated.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 }),
            Activation::Linear => Array2::ones(activated.raw_dim()),
        }
    }

    /// Get appropriate weight initialization scale
    pub fn init_scale(&self, fan_in: usize, fan_out: usize) -> f64 {
        match self {
            Activation::ReLU | Activation::LeakyReLU => {
                // He initialization
                (2.0 / fan_in as f64).sqrt()
            }
            Activation::Sigmoid | Activation::Tanh => {
                // Xavier/Glorot initialization
                (2.0 / (fan_in + fan_out) as f64).sqrt()
            }
            Activation::Linear => {
                // Xavier for linear
                (1.0 / fan_in as f64).sqrt()
            }
        }
    }
}

impl From<&str> for Activation {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "relu" => Activation::ReLU,
            "leaky_relu" | "leakyrelu" => Activation::LeakyReLU,
            "linear" | "none" => Activation::Linear,
            _ => panic!("Unknown activation function: {}", s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn sigmoid_works() {
        let z = array![[0.0, 1.0, -1.0]];
        let out = Activation::Sigmoid.apply(z);
        assert_abs_diff_eq!(out[[0, 0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(out[[0, 1]], 0.7310585, epsilon = 1e-6);
    }

    #[test]
    fn relu_works() {
        let z = array![[-1.0, 0.0, 1.0]];
        let out = Activation::ReLU.apply(z);
        assert_eq!(out, array![[0.0, 0.0, 1.0]]);
    }
}
