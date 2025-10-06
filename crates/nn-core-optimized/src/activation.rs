//! Activation helpers tuned for tight loops.

#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid_derivative_from_output(output: f64) -> f64 {
    output * (1.0 - output)
}
