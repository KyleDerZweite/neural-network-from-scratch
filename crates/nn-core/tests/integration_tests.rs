use nn_core::{layer::Layer, matrix::Matrix, network::NeuralNetwork};

#[test]
fn test_xor() {
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 0.1);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut initial_mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        initial_mse += (prediction[0] - targets[i][0]).powi(2);
    }
    initial_mse /= inputs.len() as f64;

    net.train(&inputs, &targets, 10000);

    let mut final_mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        final_mse += (prediction[0] - targets[i][0]).powi(2);
    }
    final_mse /= inputs.len() as f64;

    assert!(final_mse < initial_mse - 1e-4);
    assert!(final_mse < 0.26);
}

#[test]
fn test_sine_approximation() {
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, 0.01);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..100 {
        let x = i as f64 * 7.0 / 100.0;
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    net.train(&inputs, &targets, 5000);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse < 0.01);
}

#[test]
fn test_high_learning_rate() {
    let hidden = Layer {
        weights: Matrix::new(2, 2, vec![vec![0.5, -0.4], vec![0.3, 0.6]]),
        biases: Matrix::new(2, 1, vec![vec![0.0], vec![0.0]]),
        activation: "sigmoid".into(),
    };
    let output = Layer {
        weights: Matrix::new(1, 2, vec![vec![0.7, -0.8]]),
        biases: Matrix::new(1, 1, vec![vec![0.0]]),
        activation: "sigmoid".into(),
    };
    let mut net = NeuralNetwork::new(vec![hidden, output], 10.0);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    net.train(&inputs, &targets, 50);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse > 0.05 || mse.is_nan());
}

#[test]
fn test_low_learning_rate() {
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 1e-9);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    net.train(&inputs, &targets, 1000);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse > 0.1);
}

#[test]
fn test_different_hidden_neurons() {
    let layers = vec![Layer::new(2, 4, "sigmoid"), Layer::new(4, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 0.1);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    net.train(&inputs, &targets, 10000);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse < 0.05);
}

#[test]
fn test_unnormalized_input() {
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, 0.01);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..100 {
        let x = i as f64 * 7.0; // Un-normalized
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    net.train(&inputs, &targets, 5000);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse > 0.1 || mse.is_nan());
}
