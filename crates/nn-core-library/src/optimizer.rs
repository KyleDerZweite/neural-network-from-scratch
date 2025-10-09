//! Optimisation algorithms with proper implementation.

use ndarray::{Array2, Zip};

/// Available optimizers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerKind {
    /// Stochastic gradient descent with optional momentum.
    Sgd { momentum: f64 },
    /// Adam optimizer with bias correction.
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// RMSprop optimizer.
    RMSprop { beta: f64, epsilon: f64 },
}

impl Default for OptimizerKind {
    fn default() -> Self {
        OptimizerKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl OptimizerKind {
    pub fn sgd() -> Self {
        OptimizerKind::Sgd { momentum: 0.0 }
    }

    pub fn sgd_momentum(momentum: f64) -> Self {
        OptimizerKind::Sgd { momentum }
    }

    pub fn adam() -> Self {
        OptimizerKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    pub fn rmsprop() -> Self {
        OptimizerKind::RMSprop {
            beta: 0.9,
            epsilon: 1e-8,
        }
    }
}

/// Internal optimizer state tracking.
pub enum OptimizerState {
    Sgd(SgdState),
    Adam(AdamState),
    RMSprop(RMSpropState),
}

impl OptimizerState {
    pub fn new(kind: OptimizerKind, layer_shapes: &[(usize, usize)]) -> Self {
        match kind {
            OptimizerKind::Sgd { momentum } => {
                OptimizerState::Sgd(SgdState::new(momentum, layer_shapes))
            }
            OptimizerKind::Adam {
                beta1,
                beta2,
                epsilon,
            } => OptimizerState::Adam(AdamState::new(beta1, beta2, epsilon, layer_shapes)),
            OptimizerKind::RMSprop { beta, epsilon } => {
                OptimizerState::RMSprop(RMSpropState::new(beta, epsilon, layer_shapes))
            }
        }
    }

    /// Call this once per training step (after all layers updated)
    pub fn step(&mut self) {
        match self {
            OptimizerState::Adam(state) => state.timestep += 1,
            _ => {}
        }
    }

    pub fn apply_layer(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        match self {
            OptimizerState::Sgd(state) => state.apply(
                layer_index,
                weight_grad,
                bias_grad,
                weights,
                biases,
                learning_rate,
            ),
            OptimizerState::Adam(state) => state.apply(
                layer_index,
                weight_grad,
                bias_grad,
                weights,
                biases,
                learning_rate,
            ),
            OptimizerState::RMSprop(state) => state.apply(
                layer_index,
                weight_grad,
                bias_grad,
                weights,
                biases,
                learning_rate,
            ),
        }
    }
}

/// SGD with optional momentum
pub struct SgdState {
    momentum: f64,
    velocity_w: Vec<Array2<f64>>,
    velocity_b: Vec<Array2<f64>>,
}

impl SgdState {
    pub fn new(momentum: f64, layer_shapes: &[(usize, usize)]) -> Self {
        let mut velocity_w = Vec::new();
        let mut velocity_b = Vec::new();

        for &(output_size, input_size) in layer_shapes {
            velocity_w.push(Array2::zeros((output_size, input_size)));
            velocity_b.push(Array2::zeros((output_size, 1)));
        }

        Self {
            momentum,
            velocity_w,
            velocity_b,
        }
    }

    fn apply(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        if self.momentum > 0.0 {
            // SGD with momentum
            let vw = &mut self.velocity_w[layer_index];
            let vb = &mut self.velocity_b[layer_index];

            Zip::from(&mut *vw)
                .and(weight_grad)
                .for_each(|v, &grad| *v = self.momentum * *v + learning_rate * grad);

            Zip::from(&mut *vb)
                .and(bias_grad)
                .for_each(|v, &grad| *v = self.momentum * *v + learning_rate * grad);

            Zip::from(weights).and(&*vw).for_each(|w, &v| *w -= v);
            Zip::from(biases).and(&*vb).for_each(|b, &v| *b -= v);
        } else {
            // Vanilla SGD
            Zip::from(weights)
                .and(weight_grad)
                .for_each(|w, &grad| *w -= learning_rate * grad);

            Zip::from(biases)
                .and(bias_grad)
                .for_each(|b, &grad| *b -= learning_rate * grad);
        }
    }
}

/// Adam optimizer with FIXED timestep management
pub struct AdamState {
    timestep: usize,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m_w: Vec<Array2<f64>>,
    v_w: Vec<Array2<f64>>,
    m_b: Vec<Array2<f64>>,
    v_b: Vec<Array2<f64>>,
}

impl AdamState {
    pub fn new(beta1: f64, beta2: f64, epsilon: f64, layer_shapes: &[(usize, usize)]) -> Self {
        let mut m_w = Vec::new();
        let mut v_w = Vec::new();
        let mut m_b = Vec::new();
        let mut v_b = Vec::new();

        for &(output_size, input_size) in layer_shapes {
            m_w.push(Array2::zeros((output_size, input_size)));
            v_w.push(Array2::zeros((output_size, input_size)));
            m_b.push(Array2::zeros((output_size, 1)));
            v_b.push(Array2::zeros((output_size, 1)));
        }

        Self {
            timestep: 0,
            beta1,
            beta2,
            epsilon,
            m_w,
            v_w,
            m_b,
            v_b,
        }
    }

    fn apply(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        // Use current timestep (will be incremented after all layers)
        let t = (self.timestep + 1) as f64;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);

        let m_w = &mut self.m_w[layer_index];
        let v_w = &mut self.v_w[layer_index];
        let m_b = &mut self.m_b[layer_index];
        let v_b = &mut self.v_b[layer_index];

        // Update first moment (momentum)
        Zip::from(&mut *m_w)
            .and(weight_grad)
            .for_each(|m, &grad| *m = beta1 * *m + (1.0 - beta1) * grad);
        Zip::from(&mut *m_b)
            .and(bias_grad)
            .for_each(|m, &grad| *m = beta1 * *m + (1.0 - beta1) * grad);

        // Update second moment (variance)
        Zip::from(&mut *v_w)
            .and(weight_grad)
            .for_each(|v, &grad| *v = beta2 * *v + (1.0 - beta2) * grad * grad);
        Zip::from(&mut *v_b)
            .and(bias_grad)
            .for_each(|v, &grad| *v = beta2 * *v + (1.0 - beta2) * grad * grad);

        // Update parameters
        Zip::from(weights)
            .and(&*m_w)
            .and(&*v_w)
            .for_each(|w, m, v| {
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                *w -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            });

        Zip::from(biases).and(&*m_b).and(&*v_b).for_each(|b, m, v| {
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *b -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        });
    }
}

/// RMSprop optimizer
pub struct RMSpropState {
    beta: f64,
    epsilon: f64,
    v_w: Vec<Array2<f64>>,
    v_b: Vec<Array2<f64>>,
}

impl RMSpropState {
    pub fn new(beta: f64, epsilon: f64, layer_shapes: &[(usize, usize)]) -> Self {
        let mut v_w = Vec::new();
        let mut v_b = Vec::new();

        for &(output_size, input_size) in layer_shapes {
            v_w.push(Array2::zeros((output_size, input_size)));
            v_b.push(Array2::zeros((output_size, 1)));
        }

        Self {
            beta,
            epsilon,
            v_w,
            v_b,
        }
    }

    fn apply(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        let beta = self.beta;
        let epsilon = self.epsilon;

        let v_w = &mut self.v_w[layer_index];
        let v_b = &mut self.v_b[layer_index];

        // Update moving average of squared gradients
        Zip::from(&mut *v_w)
            .and(weight_grad)
            .for_each(|v, &grad| *v = beta * *v + (1.0 - beta) * grad * grad);
        Zip::from(&mut *v_b)
            .and(bias_grad)
            .for_each(|v, &grad| *v = beta * *v + (1.0 - beta) * grad * grad);

        // Update parameters
        Zip::from(weights)
            .and(&*v_w)
            .and(weight_grad)
            .for_each(|w, v, &grad| *w -= learning_rate * grad / (v.sqrt() + epsilon));

        Zip::from(biases)
            .and(&*v_b)
            .and(bias_grad)
            .for_each(|b, v, &grad| *b -= learning_rate * grad / (v.sqrt() + epsilon));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sgd_descends_along_gradient() {
        let layer_shapes = vec![(1, 1)];
        let mut state = OptimizerState::new(OptimizerKind::sgd(), &layer_shapes);

        let mut weights = array![[1.0]];
        let mut biases = array![[0.5]];
        let grad_w = array![[0.2]];
        let grad_b = array![[0.1]];

        state.apply_layer(0, &grad_w, &grad_b, &mut weights, &mut biases, 0.1);

        assert!(weights[(0, 0)] < 1.0);
        assert!(biases[(0, 0)] < 0.5);
    }

    #[test]
    fn adam_timestep_management() {
        let layer_shapes = vec![(2, 2)];
        let mut state = OptimizerState::new(OptimizerKind::adam(), &layer_shapes);

        match &state {
            OptimizerState::Adam(adam) => assert_eq!(adam.timestep, 0),
            _ => panic!("Expected Adam"),
        }

        state.step();

        match &state {
            OptimizerState::Adam(adam) => assert_eq!(adam.timestep, 1),
            _ => panic!("Expected Adam"),
        }
    }
}
