use chrono::prelude::*;
use clap::Parser;
use plotters::prelude::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    task: String,
}

fn main() {
    let args = Args::parse();

    match args.task.as_str() {
        "xor-core" => run_xor_core(),
        "xor-library" => run_xor_library(),
        "xor-optimized" => run_xor_optimized(),
        "sin-core" => run_sin_core(0.24),
        "sin-library" => run_sin_library(0.24),
        "sin-optimized" => run_sin_optimized(0.24),
        _ => println!("Invalid task. Please use 'xor-core', 'xor-library', 'xor-optimized', 'sin-core', 'sin-library', or 'sin-optimized'."),
    }
}

fn run_xor_core() {
    use nn_core::{layer::Layer, network::NeuralNetwork};
    println!("Running XOR task with nn-core...");
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 0.5);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "xor-core").unwrap();

    #[cfg(feature = "profiling")]
    report_profiling_core("xor-core");

    println!("\nXOR predictions:");
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        println!(
            "Input: {:?}, Prediction: {:.4}, Target: {}",
            inputs[i], prediction[0], targets[i][0]
        );
    }
}

fn run_xor_library() {
    use nn_core_library::{layer::Layer, network::NeuralNetwork};
    println!("Running XOR task with nn-core-library...");
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 0.5);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "xor-library").unwrap();

    // No profiling for library

    println!("\nXOR predictions:");
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        println!(
            "Input: {:?}, Prediction: {:.4}, Target: {}",
            inputs[i], prediction[0], targets[i][0]
        );
    }
}

fn run_xor_optimized() {
    use nn_core_optimized::{layer::Layer, network::NeuralNetwork};
    println!("Running XOR task with nn-core-optimized...");
    let layers = vec![Layer::new(2, 2, "sigmoid"), Layer::new(2, 1, "sigmoid")];
    let mut net = NeuralNetwork::new(layers, 0.5);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "xor-optimized").unwrap();

    // No profiling for optimized

    println!("\nXOR predictions:");
    for i in 0..inputs.len() {
        let prediction = net.predict(&inputs[i]);
        println!(
            "Input: {:?}, Prediction: {:.4}, Target: {}",
            inputs[i], prediction[0], targets[i][0]
        );
    }
}

fn run_sin_core(learning_rate: f64) {
    use nn_core::{layer::Layer, network::NeuralNetwork};
    println!("Running sine approximation task with nn-core...");
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, learning_rate);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..1000 {
        let x = i as f64 * 7.0 / 1000.0;
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "sin-core").unwrap();

    #[cfg(feature = "profiling")]
    report_profiling_core("sin-core");

    println!("\nSine approximation complete.");

    println!("\nSine approximation predictions:");
    for i in (0..10).map(|x| x * 20) {
        let input = &inputs[i];
        let target = &targets[i];
        let prediction = net.predict(input);
        println!(
            "Input: {:.2}, Prediction: {:.4}, Target: {:.4}",
            input[0], prediction[0], target[0]
        );
    }
    
    let min_loss = *loss_history.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Minimum loss achieved: {}", min_loss);
}

fn run_sin_library(learning_rate: f64) {
    use nn_core_library::{layer::Layer, network::NeuralNetwork};
    println!("Running sine approximation task with nn-core-library...");
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, learning_rate);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..1000 {
        let x = i as f64 * 7.0 / 1000.0;
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "sin-library").unwrap();

    println!("\nSine approximation complete.");

    println!("\nSine approximation predictions:");
    for i in (0..10).map(|x| x * 20) {
        let input = &inputs[i];
        let target = &targets[i];
        let prediction = net.predict(input);
        println!(
            "Input: {:.2}, Prediction: {:.4}, Target: {:.4}",
            input[0], prediction[0], target[0]
        );
    }
    
    let min_loss = *loss_history.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Minimum loss achieved: {}", min_loss);
}

fn run_sin_optimized(learning_rate: f64) {
    use nn_core_optimized::{layer::Layer, network::NeuralNetwork};
    println!("Running sine approximation task with nn-core-optimized...");
    let layers = vec![Layer::new(1, 32, "sigmoid"), Layer::new(32, 1, "linear")];
    let mut net = NeuralNetwork::new(layers, learning_rate);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..1000 {
        let x = i as f64 * 7.0 / 1000.0;
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }

    let loss_history = net.train(&inputs, &targets, 20000);
    plot_loss(&loss_history, "sin-optimized").unwrap();

    println!("\nSine approximation complete.");

    println!("\nSine approximation predictions:");
    for i in (0..10).map(|x| x * 20) {
        let input = &inputs[i];
        let target = &targets[i];
        let prediction = net.predict(input);
        println!(
            "Input: {:.2}, Prediction: {:.4}, Target: {:.4}",
            input[0], prediction[0], target[0]
        );
    }
    
    let min_loss = *loss_history.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Minimum loss achieved: {}", min_loss);
}

fn plot_loss(loss_history: &Vec<f64>, task_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let now = Utc::now();
    let filename = format!(
        "plots/{}_{}.png",
        task_name,
        now.format("%Y-%m-%d_%H-%M-%S")
    );
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = loss_history.iter().cloned().fold(0.0, f64::max);
    let min_loss = loss_history
        .iter()
        .cloned()
        .filter(|&l| l > 0.0)
        .fold(f64::INFINITY, f64::min);

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss History", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..loss_history.len(), (min_loss..max_loss).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Epoch (x1000)")
        .y_desc("Mean Squared Error (MSE) - Log Scale")
        .draw()?;

    chart.draw_series(LineSeries::new(
        loss_history.iter().enumerate().map(|(i, &loss)| (i, loss)),
        &RED,
    ))?;

    root.present()?;
    println!("Loss history plot saved to {}", filename);

    Ok(())
}