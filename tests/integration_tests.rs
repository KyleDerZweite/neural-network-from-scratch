use nn_core::layer::Layer;
use nn_core::network::NeuralNetwork;

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

    net.train(&inputs, &targets, 50000);

    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        assert!((prediction[0] - targets[i][0]).abs() < 0.1);
    }
}

#[test]
fn test_sine_approximation() {
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, 0.01);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..200 {
        let x = i as f64 * 7.0 / 200.0;
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    net.train(&inputs, &targets, 20000);

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
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 10.0);

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

    assert!(mse > 0.1 || mse.is_nan());
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

    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        assert!((prediction[0] - targets[i][0]).abs() < 0.1);
    }
}

#[test]
fn test_unnormalized_input() {
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, 0.01);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..200 {
        let x = i as f64 * 7.0; // Un-normalized
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    net.train(&inputs, &targets, 20000);

    let mut mse = 0.0;
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        mse += (prediction[0] - targets[i][0]).powi(2);
    }
    mse /= inputs.len() as f64;

    assert!(mse > 0.1 || mse.is_nan());
}