//! Optimisation algorithms for neural network training.

use ndarray::{Array2, Zip};

/// Available optimisers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    /// Vanilla stochastic gradient descent.
    Sgd,
    /// Adam optimiser with bias correction.
    Adam,
}

impl Default for OptimizerKind {
    fn default() -> Self {
        OptimizerKind::Adam
    }
}

/// Internal optimiser state tracking.
pub enum OptimizerState {
    Sgd,
    Adam(AdamState),
}

impl OptimizerState {
    pub fn new(kind: OptimizerKind, layer_shapes: &[(usize, usize, usize, usize)]) -> Self {
        match kind {
            OptimizerKind::Sgd => OptimizerState::Sgd,
            OptimizerKind::Adam => OptimizerState::Adam(AdamState::new(layer_shapes)),
        }
    }

    pub fn apply(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        match self {
            OptimizerState::Sgd => apply_sgd(weights, biases, weight_grad, bias_grad, learning_rate),
            OptimizerState::Adam(state) => state.apply(layer_index, weight_grad, bias_grad, weights, biases, learning_rate),
        }
    }
}

fn apply_sgd(
    weights: &mut Array2<f64>,
    biases: &mut Array2<f64>,
    weight_grad: &Array2<f64>,
    bias_grad: &Array2<f64>,
    learning_rate: f64,
) {
    Zip::from(weights)
        .and(weight_grad)
        .for_each(|w, &grad| *w -= learning_rate * grad);

    Zip::from(biases)
        .and(bias_grad)
        .for_each(|b, &grad| *b -= learning_rate * grad);
}

/// Adam optimiser state.
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
    pub fn new(layer_shapes: &[(usize, usize, usize, usize)]) -> Self {
        let mut m_w = Vec::new();
        let mut v_w = Vec::new();
        let mut m_b = Vec::new();
        let mut v_b = Vec::new();

        for &(w_rows, w_cols, b_rows, b_cols) in layer_shapes {
            m_w.push(Array2::zeros((w_rows, w_cols)));
            v_w.push(Array2::zeros((w_rows, w_cols)));
            m_b.push(Array2::zeros((b_rows, b_cols)));
            v_b.push(Array2::zeros((b_rows, b_cols)));
        }

        Self {
            timestep: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_w,
            v_w,
            m_b,
            v_b,
        }
    }

    pub fn apply(
        &mut self,
        layer_index: usize,
        weight_grad: &Array2<f64>,
        bias_grad: &Array2<f64>,
        weights: &mut Array2<f64>,
        biases: &mut Array2<f64>,
        learning_rate: f64,
    ) {
        self.timestep += 1;
        let timestep_f = self.timestep as f64;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;

        let m_w = &mut self.m_w[layer_index];
        let v_w = &mut self.v_w[layer_index];
        let m_b = &mut self.m_b[layer_index];
        let v_b = &mut self.v_b[layer_index];

        Zip::from(m_w)
            .and(weight_grad)
            .for_each(|m, &grad| *m = beta1 * *m + (1.0 - beta1) * grad);
        Zip::from(v_w)
            .and(weight_grad)
            .for_each(|v, &grad| *v = beta2 * *v + (1.0 - beta2) * grad * grad);

        Zip::from(m_b)
            .and(bias_grad)
            .for_each(|m, &grad| *m = beta1 * *m + (1.0 - beta1) * grad);
        Zip::from(v_b)
            .and(bias_grad)
            .for_each(|v, &grad| *v = beta2 * *v + (1.0 - beta2) * grad * grad);

        let bias_correction1 = 1.0 - beta1.powf(timestep_f);
        let bias_correction2 = 1.0 - beta2.powf(timestep_f);

        Zip::from(weights)
            .and(m_w)
            .and(v_w)
            .for_each(|w, &m, &v| {
                let m_hat = m / bias_correction1;
                let v_hat = v / bias_correction2;
                *w -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            });

        Zip::from(biases)
            .and(m_b)
            .and(v_b)
            .for_each(|b, &m, &v| {
                let m_hat = m / bias_correction1;
                let v_hat = v / bias_correction2;
                *b -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sgd_descends_along_gradient() {
        let layer_shapes = vec![(1, 1, 1, 1)];
        let mut state = OptimizerState::new(OptimizerKind::Sgd, &layer_shapes);

        let mut weights = array![[1.0]];
        let mut biases = array![[0.5]];
        let grad_w = array![[0.2]];
        let grad_b = array![[0.1]];

        state.apply(0, &grad_w, &grad_b, &mut weights, &mut biases, 0.1);

        assert!(weights[(0, 0)] < 1.0);
        assert!(biases[(0, 0)] < 0.5);
    }
}
