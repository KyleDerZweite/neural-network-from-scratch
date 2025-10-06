//! Feed-forward training loop powered by ndarray with optional GPU acceleration
//! and configurable optimisation strategies.

use crate::layer::Layer;
use crate::optimizer::{OptimizerKind, OptimizerState};
use ndarray::{Array2, Axis};

#[cfg(feature = "gpu")]
use crate::gpu::{GpuAccelerator, DEFAULT_GPU_WORKLOAD_THRESHOLD};

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    optimizer_kind: OptimizerKind,
    optimizer_state: OptimizerState,
    #[cfg(feature = "gpu")]
    gpu: Option<GpuAccelerator>,
    #[cfg(feature = "gpu")]
    gpu_workload_threshold: usize,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self::with_optimizer(layers, learning_rate, OptimizerKind::default())
    }

    pub fn with_optimizer(
        layers: Vec<Layer>,
        learning_rate: f64,
        optimizer_kind: OptimizerKind,
    ) -> Self {
        let layer_shapes: Vec<_> = layers
            .iter()
            .map(|layer| (layer.weights.nrows(), layer.weights.ncols()))
            .collect();

        let optimizer_state = OptimizerState::new(optimizer_kind, &layer_shapes);

        #[cfg(feature = "gpu")]
        let gpu = match GpuAccelerator::new() {
            Ok(accel) => Some(accel),
            Err(err) => {
                log::warn!("GPU accelerator unavailable, continuing with CPU: {err}");
                None
            }
        };

        Self {
            layers,
            learning_rate,
            optimizer_kind,
            optimizer_state,
            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            gpu_workload_threshold: DEFAULT_GPU_WORKLOAD_THRESHOLD,
        }
    }

    pub fn optimizer_kind(&self) -> OptimizerKind {
        self.optimizer_kind
    }

    #[cfg(feature = "gpu")]
    pub fn set_gpu_workload_threshold(&mut self, threshold: usize) {
        self.gpu_workload_threshold = threshold.max(1);
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

                let mut activations: Vec<Array2<f64>> = Vec::with_capacity(self.layers.len() + 1);
                activations.push(input);
                for layer in &self.layers {
                    let weighted = self.matmul(&layer.weights, activations.last().expect("activation available"));
                    let z = weighted + &layer.biases;
                    let activation = layer.activation.apply(z);
                    activations.push(activation);
                }

                let output_activation = activations.last().expect("output activation available");
                let mut delta = output_activation - &target;

                // Apply output layer activation derivative
                let last_layer = self.layers.last().unwrap();
                let output_deriv = last_layer.activation.derivative(output_activation);
                delta = delta * &output_deriv;

                for layer_idx in (0..self.layers.len()).rev() {
                    crate::profile_scope!("network.backward");
                    let activation_prev = &activations[layer_idx];
                    let activation_prev_t = activation_prev.t().to_owned();
                    let weight_grad = self.matmul(&delta, &activation_prev_t);
                    let bias_grad = delta.clone();

                    let next_delta = if layer_idx > 0 {
                        let weights_t = self.layers[layer_idx].weights.t().to_owned();
                        let mut propagated = self.matmul(&weights_t, &delta);
                        // Apply previous layer's activation derivative
                        let prev_deriv = self.layers[layer_idx - 1].activation.derivative(&activations[layer_idx]);
                        propagated = propagated * prev_deriv;
                        Some(propagated)
                    } else {
                        None
                    };

                    {
                        let layer = &mut self.layers[layer_idx];
                        self.optimizer_state.apply_layer(
                            layer_idx,
                            &weight_grad,
                            &bias_grad,
                            &mut layer.weights,
                            &mut layer.biases,
                            self.learning_rate,
                        );
                    }

                    if let Some(propagated) = next_delta {
                        delta = propagated;
                    }
                }

                // Step the optimizer (for Adam timestep management)
                self.optimizer_state.step();
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
            let weighted = self.matmul(&layer.weights, &current) + &layer.biases;
            current = layer.activation.apply(weighted);
        }
        current
            .index_axis(Axis(1), 0)
            .to_owned()
            .into_raw_vec()
    }

    fn matmul(&self, lhs: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            let (m, k_lhs) = lhs.dim();
            let (k_rhs, n) = rhs.dim();
            debug_assert_eq!(k_lhs, k_rhs, "Matrix dimensions mismatch");

            if gpu.is_workload_worth_gpu(m, n, k_lhs, self.gpu_workload_threshold) {
                match gpu.matmul(lhs, rhs) {
                    Ok(result) => return result,
                    Err(err) => log::warn!("GPU matmul failed, falling back to CPU: {err}"),
                }
            }
        }

        lhs.dot(rhs)
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
