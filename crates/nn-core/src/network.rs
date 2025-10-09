//! A feed-forward neural network.

use crate::activation;
use crate::layer::Layer;
use crate::matrix::Matrix;

/// A feed-forward neural network.
pub struct NeuralNetwork {
    /// The layers of the network.
    pub layers: Vec<Layer>,
    /// The learning rate of the network.
    pub learning_rate: f64,
}

impl NeuralNetwork {
    /// Creates a new neural network.
    ///
    /// # Arguments
    ///
    /// * `layers` - The layers of the network.
    /// * `learning_rate` - The learning rate of the network.
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }

    /// Trains the neural network.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The training inputs.
    /// * `targets` - The training targets.
    /// * `epochs` - The number of epochs to train for.
    pub fn train(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &Vec<Vec<f64>>,
        epochs: usize,
    ) -> Vec<f64> {
        let mut loss_history = Vec::new();
        for i in 0..epochs {
            for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
                crate::profile_scope!("network.train.step");
                let input = Matrix::from_vec(input_vec);
                let target = Matrix::from_vec(target_vec);

                // Forward pass
                let mut activations: Vec<Matrix> = vec![input];
                {
                    crate::profile_scope!("network.forward");
                    for layer in &self.layers {
                        let z = layer
                            .weights
                            .mul(activations.last().unwrap())
                            .add(&layer.biases);
                        let activation = if layer.activation == "sigmoid" {
                            z.map(activation::sigmoid)
                        } else {
                            z // linear activation
                        };
                        activations.push(activation);
                    }
                }

                // Backward pass
                let output = activations.last().unwrap();
                let error = output.subtract(&target);
                let mut delta = if self.layers.last().unwrap().activation == "sigmoid" {
                    error.dot(&output.map(activation::sigmoid_derivative_from_output))
                } else {
                    error
                };

                {
                    crate::profile_scope!("network.backward");
                    for l in (0..self.layers.len()).rev() {
                        let d_w = delta.mul_transpose(&activations[l]);
                        let d_b = delta.clone();

                        let weight_update = d_w.map(|x| -self.learning_rate * x);
                        let bias_update = d_b.map(|x| -self.learning_rate * x);

                        let next_delta = if l > 0 {
                            let mut propagated = self.layers[l].weights.transpose_mul(&delta);
                            if self.layers[l - 1].activation == "sigmoid" {
                                propagated = propagated.dot(
                                    &activations[l].map(activation::sigmoid_derivative_from_output),
                                );
                            }
                            Some(propagated)
                        } else {
                            None
                        };

                        self.layers[l].weights = self.layers[l].weights.add(&weight_update);
                        self.layers[l].biases = self.layers[l].biases.add(&bias_update);

                        if let Some(propagated_delta) = next_delta {
                            delta = propagated_delta;
                        }
                    }
                }
            }

            if i % 1000 == 0 {
                let mut mse = 0.0;
                for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
                    let prediction = self.predict(input_vec);
                    let target = target_vec[0];
                    mse += (prediction[0] - target).powi(2);
                }
                mse /= inputs.len() as f64;
                println!("Epoch: {}, MSE: {}", i, mse);
                loss_history.push(mse);
            }
        }
        loss_history
    }

    /// Makes a prediction with the neural network.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to make a prediction for.
    pub fn predict(&self, input: &Vec<f64>) -> Vec<f64> {
        crate::profile_scope!("network.predict");
        let mut current_input = Matrix::from_vec(input);

        for layer in &self.layers {
            let mut output = layer.weights.mul(&current_input);
            output = output.add(&layer.biases);

            if layer.activation == "sigmoid" {
                output = output.map(activation::sigmoid);
            }
            // No activation for "linear"

            current_input = output;
        }

        current_input.data.into_iter().flatten().collect()
    }

    /// Gets a reference to the layers.
    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }
}
