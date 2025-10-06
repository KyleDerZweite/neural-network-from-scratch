//! Feed-forward network built on the optimised primitives.

use crate::activation;
use crate::layer::{ActivationKind, Layer};
use crate::matrix::Matrix;

/// A simple fully-connected feed-forward network.
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
    ) -> Vec<f64> {
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
                crate::profile_scope!("network.train.step");
                let input = Matrix::from_vec(input_vec);
                let target = Matrix::from_vec(target_vec);

                // forward
                let mut activations: Vec<Matrix> = vec![input];
                for layer in &self.layers {
                    let z = layer.weights.mul(activations.last().unwrap()).add(&layer.biases);
                    let activation = match layer.activation {
                        ActivationKind::Sigmoid => z.map(activation::sigmoid),
                        ActivationKind::Linear => z,
                    };
                    activations.push(activation);
                }

                // backward
                let output = activations.last().unwrap();
                let mut delta = output.subtract(&target);
                if matches!(self.layers.last().map(|l| &l.activation), Some(ActivationKind::Sigmoid)) {
                    delta = delta.dot(&output.map(activation::sigmoid_derivative_from_output));
                }

                for layer_idx in (0..self.layers.len()).rev() {
                    crate::profile_scope!("network.backward");
                    let d_w = delta.mul_transpose(&activations[layer_idx]);
                    let d_b = delta.clone();

                    let weight_update = d_w.map(|x| -self.learning_rate * x);
                    let bias_update = d_b.map(|x| -self.learning_rate * x);

                    let next_delta = if layer_idx > 0 {
                        let mut propagated = self.layers[layer_idx].weights.transpose_mul(&delta);
                        if matches!(self.layers[layer_idx - 1].activation, ActivationKind::Sigmoid) {
                            propagated = propagated.dot(
                                &activations[layer_idx]
                                    .map(activation::sigmoid_derivative_from_output),
                            );
                        }
                        Some(propagated)
                    } else {
                        None
                    };

                    self.layers[layer_idx].weights = self.layers[layer_idx]
                        .weights
                        .add(&weight_update);
                    self.layers[layer_idx].biases = self.layers[layer_idx]
                        .biases
                        .add(&bias_update);

                    if let Some(prop) = next_delta {
                        delta = prop;
                    }
                }
            }

            if epoch % 1_000 == 0 {
                let mut mse = 0.0;
                for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
                    let prediction = self.predict(input_vec);
                    let target = target_vec[0];
                    mse += (prediction[0] - target).powi(2);
                }
                mse /= inputs.len() as f64;
                println!("Epoch: {epoch}, MSE: {mse}");
                loss_history.push(mse);
            }
        }

        loss_history
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        crate::profile_scope!("network.predict");
        let mut current = Matrix::from_vec(input);

        for layer in &self.layers {
            let mut output = layer.weights.mul(&current);
            output = output.add(&layer.biases);
            if matches!(layer.activation, ActivationKind::Sigmoid) {
                output = output.map(activation::sigmoid);
            }
            current = output;
        }

        current.to_column_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn trains_towards_simple_target() {
        let mut network = NeuralNetwork::new(
            vec![
                Layer::new(1, 2, ActivationKind::Sigmoid),
                Layer::new(2, 1, ActivationKind::Linear),
            ],
            0.1,
        );

        let inputs = vec![vec![0.0], vec![1.0]];
        let targets = vec![vec![0.0], vec![1.0]];

    let history = network.train(&inputs, &targets, 2_000);
        assert!(!history.is_empty());

        let pred_zero = network.predict(&[0.0])[0];
        let pred_one = network.predict(&[1.0])[0];

        assert!(pred_zero < pred_one);
    }
}
