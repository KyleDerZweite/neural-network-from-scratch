//! Feed-forward training loop powered by ndarray.

use crate::activation::{self, Activation};
use crate::layer::Layer;
use ndarray::{Array2, Axis};

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
                let input = column_from(input_vec);
                let target = column_from(target_vec);

                let mut activations: Vec<Array2<f64>> = vec![input];
                for layer in &self.layers {
                    let z = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
                    let activation = layer.activation.apply(z);
                    activations.push(activation);
                }

                let output = activations.last().expect("output activation available");
                let mut delta = output - &target;

                if matches!(self.layers.last().map(|l| &l.activation), Some(Activation::Sigmoid)) {
                    delta = delta * activation::sigmoid_derivative_from_output(output);
                }

                for layer_idx in (0..self.layers.len()).rev() {
                    crate::profile_scope!("network.backward");
                    let activation_prev = &activations[layer_idx];
                    let weight_grad = delta.dot(&activation_prev.t());
                    let bias_grad = delta.clone();

                    let next_delta = if layer_idx > 0 {
                        let mut propagated = self.layers[layer_idx].weights.t().dot(&delta);
                        if matches!(self.layers[layer_idx - 1].activation, Activation::Sigmoid) {
                            let deriv = activation::sigmoid_derivative_from_output(&activations[layer_idx]);
                            propagated = propagated * deriv;
                        }
                        Some(propagated)
                    } else {
                        None
                    };

                    self.layers[layer_idx].weights =
                        &self.layers[layer_idx].weights - &(weight_grad * self.learning_rate);
                    self.layers[layer_idx].biases =
                        &self.layers[layer_idx].biases - &(bias_grad * self.learning_rate);

                    if let Some(propagated) = next_delta {
                        delta = propagated;
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
        let mut current = column_from(input);
        for layer in &self.layers {
            current = layer.weights.dot(&current) + &layer.biases;
            current = layer.activation.apply(current);
        }
        current
            .index_axis(Axis(1), 0)
            .to_owned()
            .into_raw_vec()
    }
}

fn column_from(data: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((data.len(), 1), data.to_vec())
        .expect("valid column shape for vector")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn trains_on_identity_mapping() {
        let mut nn = NeuralNetwork::new(
            vec![
                Layer::new(1, 2, Activation::Sigmoid),
                Layer::new(2, 1, Activation::Linear),
            ],
            0.1,
        );

        let inputs = vec![vec![0.0], vec![1.0]];
        let targets = vec![vec![0.0], vec![1.0]];

    let history = nn.train(&inputs, &targets, 2_000);
        assert!(!history.is_empty());

        let pred_zero = nn.predict(&[0.0])[0];
        let pred_one = nn.predict(&[1.0])[0];

    assert!(pred_zero < pred_one);
    }
}
