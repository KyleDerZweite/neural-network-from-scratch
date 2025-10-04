use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};
use meval::{Error as MevalError, Expr};
use nn_core::{activation, layer::Layer, network::NeuralNetwork};
use tokio::task;
use uuid::Uuid;

use crate::models::{
    PredictRequest, PredictResponse, StopTrainingResponse, TrainRequest, TrainResponse,
    TrainingJob, TrainingProgressResponse, TrainingStatus,
};

pub struct AppState {
    pub training_jobs: Mutex<HashMap<Uuid, TrainingJob>>,
    pub trained_networks: Mutex<HashMap<Uuid, NeuralNetwork>>,
}

const MAX_LAYERS: usize = 10;
const MAX_NEURONS: usize = 256;
const WARNING_LAYER_THRESHOLD: usize = 5;
const WARNING_NEURON_THRESHOLD: usize = 128;
const MAX_EPOCHS: usize = 500_000;
const FUNCTION_TIMEOUT: Duration = Duration::from_millis(100);

pub async fn train(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<TrainRequest>,
) -> (StatusCode, Json<TrainResponse>) {
    let job_id = Uuid::new_v4();

    // Validate network configuration constraints
    let warning = match validate_network_config(&payload.network_config) {
        Ok(warning) => warning,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(TrainResponse {
                    job_id: Uuid::nil(),
                    status: TrainingStatus::Failed,
                    warning: None,
                    message: Some(message),
                }),
            );
        }
    };

    // Validate target configuration
    if let Err(message) = validate_target(&payload.target_type, &payload.target_expression) {
        return (
            StatusCode::BAD_REQUEST,
            Json(TrainResponse {
                job_id: Uuid::nil(),
                status: TrainingStatus::Failed,
                warning: None,
                message: Some(message),
            }),
        );
    }

    let update_interval = default_update_interval(payload.network_config.epochs);

    let training_job = TrainingJob {
        id: job_id,
        network_config: payload.network_config.clone(),
        target_type: payload.target_type.clone(),
        target_expression: payload.target_expression.clone(),
        status: TrainingStatus::Pending,
        progress: 0.0,
        current_loss: None,
        loss_history: Vec::new(),
        update_frequency: Some(update_interval),
        warning: warning.clone(),
        message: None,
    };

    app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned")
        .insert(job_id, training_job);

    // Spawn training task
    let app_state_clone = Arc::clone(&app_state);
    tokio::spawn(async move {
        let result = task::spawn_blocking(move || {
            execute_training(job_id, app_state_clone, update_interval)
        })
        .await;

        if let Err(error) = result {
            eprintln!("Training task join error: {}", error);
        }
    });

    (
        StatusCode::OK,
        Json(TrainResponse {
            job_id,
            status: TrainingStatus::Pending,
            warning,
            message: None,
        }),
    )
}

pub async fn get_training_progress(
    State(app_state): State<Arc<AppState>>,
    Path(job_id): Path<Uuid>,
) -> (StatusCode, Json<TrainingProgressResponse>) {
    let jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");

    if let Some(job) = jobs.get(&job_id) {
        (
            StatusCode::OK,
            Json(TrainingProgressResponse {
                job_id: job.id,
                status: job.status.clone(),
                progress: job.progress,
                current_loss: job.current_loss,
                loss_history: job.loss_history.clone(),
                update_frequency: job.update_frequency,
                warning: job.warning.clone(),
                message: job.message.clone(),
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(TrainingProgressResponse {
                job_id,
                status: TrainingStatus::Failed,
                progress: 0.0,
                current_loss: None,
                loss_history: Vec::new(),
                update_frequency: None,
                warning: None,
                message: Some("Training job not found".to_string()),
            }),
        )
    }
}

pub async fn predict(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    let networks = app_state
        .trained_networks
        .lock()
        .expect("trained networks mutex poisoned");

    if let Some(network) = networks.get(&payload.job_id) {
        let prediction = network.predict(&payload.input);

        let actual_result = {
            let jobs = app_state
                .training_jobs
                .lock()
                .expect("training jobs mutex poisoned");
            jobs.get(&payload.job_id)
                .and_then(|job| match job.target_type.as_str() {
                    "function" => payload
                        .input
                        .get(0)
                        .copied()
                        .map(|value| evaluate_expression(&job.target_expression, value)),
                    "logical_operation" => {
                        evaluate_logical_operation(&job.target_expression, &payload.input)
                    }
                    _ => None,
                })
        };

        (
            StatusCode::OK,
            Json(PredictResponse {
                job_id: payload.job_id,
                input: payload.input,
                prediction,
                actual_result,
                message: None,
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(PredictResponse {
                job_id: payload.job_id,
                input: payload.input,
                prediction: Vec::new(),
                actual_result: None,
                message: Some("Trained network not found".into()),
            }),
        )
    }
}

pub async fn stop_training(
    State(app_state): State<Arc<AppState>>,
    Path(job_id): Path<Uuid>,
) -> (StatusCode, Json<StopTrainingResponse>) {
    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");

    if let Some(job) = jobs.get_mut(&job_id) {
        job.status = TrainingStatus::Stopped;
        (
            StatusCode::OK,
            Json(StopTrainingResponse {
                job_id,
                status: TrainingStatus::Stopped,
                message: Some("Training stopped".into()),
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(StopTrainingResponse {
                job_id,
                status: TrainingStatus::Failed,
                message: Some("Training job not found".into()),
            }),
        )
    }
}

fn execute_training(job_id: Uuid, app_state: Arc<AppState>, update_interval: usize) {
    let (job_snapshot, inputs, targets) = {
        let mut jobs = app_state
            .training_jobs
            .lock()
            .expect("training jobs mutex poisoned");
        let job = match jobs.get_mut(&job_id) {
            Some(job) => job,
            None => return,
        };

        job.status = TrainingStatus::Running;

        let training_data = if job.target_type == "function" {
            generate_function_data(&job.target_expression, 100)
        } else {
            match generate_logical_data(&job.target_expression) {
                Some(data) => data,
                None => {
                    job.status = TrainingStatus::Failed;
                    job.message = Some("Unsupported logical operation".into());
                    return;
                }
            }
        };

        (job.clone(), training_data.0, training_data.1)
    };

    let layers = match build_layers(
        &job_snapshot.network_config.layers,
        &job_snapshot.network_config.activation_function,
    ) {
        Ok(layers) => layers,
        Err(error) => {
            update_job_status(&app_state, job_id, TrainingStatus::Failed, Some(error));
            return;
        }
    };

    if let Some(sample) = inputs.first() {
        if sample.len() != job_snapshot.network_config.layers[0] {
            update_job_status(
                &app_state,
                job_id,
                TrainingStatus::Failed,
                Some(format!(
                    "Input dimensionality {} does not match configured first layer size {}",
                    sample.len(),
                    job_snapshot.network_config.layers[0]
                )),
            );
            return;
        }
    }

    if let Some(sample) = targets.first() {
        if sample.len() != *job_snapshot.network_config.layers.last().unwrap_or(&1) {
            update_job_status(
                &app_state,
                job_id,
                TrainingStatus::Failed,
                Some("Output dimensionality does not match configured last layer size".into()),
            );
            return;
        }
    }

    let mut network = NeuralNetwork::new(layers, job_snapshot.network_config.learning_rate);

    train_with_progress(
        &mut network,
        &inputs,
        &targets,
        job_snapshot.network_config.epochs,
        update_interval,
        job_id,
        Arc::clone(&app_state),
    );

    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");
    if let Some(job) = jobs.get_mut(&job_id) {
        if !matches!(job.status, TrainingStatus::Stopped | TrainingStatus::Failed) {
            job.status = TrainingStatus::Completed;
        }
    }
    drop(jobs);

    app_state
        .trained_networks
        .lock()
        .expect("trained networks mutex poisoned")
        .insert(job_id, network);
}

fn train_with_progress(
    network: &mut NeuralNetwork,
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    epochs: usize,
    update_interval: usize,
    job_id: Uuid,
    app_state: Arc<AppState>,
) {
    for epoch in 0..epochs {
        for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
            let input = nn_core::matrix::Matrix::from_vec(input_vec);
            let target = nn_core::matrix::Matrix::from_vec(target_vec);

            let mut activations: Vec<nn_core::matrix::Matrix> = vec![input];
            for layer in &network.layers {
                let z = layer
                    .weights
                    .mul(activations.last().unwrap())
                    .add(&layer.biases);
                let activation = if layer.activation == "sigmoid" {
                    z.map(activation::sigmoid)
                } else {
                    z
                };
                activations.push(activation);
            }

            let error = target.subtract(activations.last().unwrap());
            let mut delta = if network.layers.last().unwrap().activation == "sigmoid" {
                error.dot(
                    &activations
                        .last()
                        .unwrap()
                        .map(activation::sigmoid_derivative_from_output),
                )
            } else {
                error
            };

            for layer_index in (0..network.layers.len()).rev() {
                let d_w = delta.mul_transpose(&activations[layer_index]);
                let d_b = delta.clone();

                network.layers[layer_index].weights = network.layers[layer_index]
                    .weights
                    .add(&d_w.map(|value| value * network.learning_rate));
                network.layers[layer_index].biases = network.layers[layer_index]
                    .biases
                    .add(&d_b.map(|value| value * network.learning_rate));

                if layer_index > 0 {
                    delta = network.layers[layer_index].weights.transpose_mul(&delta);
                    if network.layers[layer_index - 1].activation == "sigmoid" {
                        delta = delta.dot(
                            &activations[layer_index]
                                .map(activation::sigmoid_derivative_from_output),
                        );
                    }
                }
            }
        }

        if epoch % update_interval == 0 || epoch == epochs - 1 {
            update_training_progress(
                network,
                inputs,
                targets,
                epochs,
                epoch,
                job_id,
                Arc::clone(&app_state),
            );
            if is_job_stopped(&app_state, job_id) {
                return;
            }
        }
    }
}

fn update_training_progress(
    network: &NeuralNetwork,
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    epochs: usize,
    epoch: usize,
    job_id: Uuid,
    app_state: Arc<AppState>,
) {
    let mut mse = 0.0;
    for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input_vec);
        let target = target_vec[0];
        mse += (prediction[0] - target).powi(2);
    }
    mse /= inputs.len() as f64;

    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");

    if let Some(job) = jobs.get_mut(&job_id) {
        job.progress = (epoch as f64 / epochs as f64) * 100.0;
        job.current_loss = Some(mse);
        job.loss_history.push(mse);
    }
}

fn is_job_stopped(app_state: &Arc<AppState>, job_id: Uuid) -> bool {
    let jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");
    jobs.get(&job_id)
        .map(|job| matches!(job.status, TrainingStatus::Stopped))
        .unwrap_or(false)
}

fn update_job_status(
    app_state: &Arc<AppState>,
    job_id: Uuid,
    status: TrainingStatus,
    message: Option<String>,
) {
    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");
    if let Some(job) = jobs.get_mut(&job_id) {
        job.status = status;
        if let Some(msg) = message {
            job.message = Some(msg);
        }
    }
}

fn validate_network_config(
    config: &crate::models::NeuralNetworkConfiguration,
) -> Result<Option<String>, String> {
    if config.layers.len() < 2 {
        return Err("Network must contain at least an input and an output layer".into());
    }
    if config.layers.len() > MAX_LAYERS {
        return Err(format!("Network cannot exceed {MAX_LAYERS} layers"));
    }
    if let Some(&max_neurons) = config.layers.iter().max() {
        if max_neurons > MAX_NEURONS {
            return Err(format!("Layers cannot exceed {MAX_NEURONS} neurons"));
        }
    }
    if config.epochs == 0 {
        return Err("Epochs must be greater than zero".into());
    }
    if config.epochs > MAX_EPOCHS {
        return Err(format!("Epochs cannot exceed {MAX_EPOCHS}"));
    }

    let mut warnings = Vec::new();
    if config.layers.len() > WARNING_LAYER_THRESHOLD {
        warnings.push(format!(
            "Layer count {} is high; training may be slow",
            config.layers.len()
        ));
    }
    if config
        .layers
        .iter()
        .any(|&neurons| neurons > WARNING_NEURON_THRESHOLD)
    {
        warnings.push(
            "Neuron count exceeds 128 in at least one layer; expect longer training times".into(),
        );
    }

    Ok(if warnings.is_empty() {
        None
    } else {
        Some(warnings.join(". "))
    })
}

fn validate_target(target_type: &str, target_expression: &str) -> Result<(), String> {
    match target_type {
        "function" => validate_mathematical_expression(target_expression),
        "logical_operation" => validate_logical_operation(target_expression),
        _ => Err("Unsupported target type".into()),
    }
}

fn validate_mathematical_expression(expression: &str) -> Result<(), String> {
    if expression.trim().is_empty() {
        return Err("Expression cannot be empty".into());
    }

    let expr_string = expression.to_string();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result: Result<(), String> = (|| {
            let expr: Expr = expr_string
                .parse()
                .map_err(|err: MevalError| err.to_string())?;
            let evaluator = expr
                .bind("x")
                .map_err(|err: MevalError| err.to_string())?;
            let value = evaluator(0.0);
            if value.is_finite() {
                Ok(())
            } else {
                Err("Expression evaluation produced a non-finite value at x = 0".into())
            }
        })();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(FUNCTION_TIMEOUT) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(format!("Invalid mathematical expression: {error}")),
        Err(_) => Err("Expression validation timed out".into()),
    }
}

fn validate_logical_operation(operation: &str) -> Result<(), String> {
    match operation.to_lowercase().as_str() {
        "xor" | "and" | "or" | "not" => Ok(()),
        _ => Err("Unsupported logical operation".into()),
    }
}

fn build_layers(layer_sizes: &[usize], activation_function: &str) -> Result<Vec<Layer>, String> {
    if layer_sizes.len() < 2 {
        return Err("At least two layer sizes are required".into());
    }
    let mut layers = Vec::new();
    for index in 0..layer_sizes.len() - 1 {
        let input_size = layer_sizes[index];
        let output_size = layer_sizes[index + 1];
        let activation = if index == layer_sizes.len() - 2 {
            "linear"
        } else {
            activation_function
        };
        layers.push(Layer::new(input_size, output_size, activation));
    }
    Ok(layers)
}

fn generate_function_data(expression: &str, num_samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..num_samples {
        let x = (i as f64 / num_samples as f64) * std::f64::consts::TAU - std::f64::consts::PI;
        let y = evaluate_expression(expression, x);
        inputs.push(vec![x]);
        targets.push(vec![y]);
    }

    (inputs, targets)
}

fn generate_logical_data(operation: &str) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    match operation.to_lowercase().as_str() {
        "xor" => Some(generate_xor_data()),
        "and" => Some(generate_and_data()),
        "or" => Some(generate_or_data()),
        "not" => Some(generate_not_data()),
        _ => None,
    }
}

fn generate_xor_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    (inputs, targets)
}

fn generate_and_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
    (inputs, targets)
}

fn generate_or_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]];
    (inputs, targets)
}

fn generate_not_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec![vec![0.0], vec![1.0]];
    let targets = vec![vec![1.0], vec![0.0]];
    (inputs, targets)
}

fn evaluate_expression(expression: &str, x: f64) -> f64 {
    match expression.parse::<Expr>() {
        Ok(expr) => match expr.bind("x") {
            Ok(evaluator) => {
                let value = evaluator(x);
                if value.is_finite() {
                    value
                } else {
                    0.0
                }
            }
            Err(_) => 0.0,
        },
        Err(_) => 0.0,
    }
}

fn evaluate_logical_operation(operation: &str, input: &[f64]) -> Option<f64> {
    match operation.to_lowercase().as_str() {
        "xor" => Some(((input.get(0)? > &0.5) ^ (input.get(1)? > &0.5)) as u8 as f64),
        "and" => Some(((input.get(0)? > &0.5) && (input.get(1)? > &0.5)) as u8 as f64),
        "or" => Some(((input.get(0)? > &0.5) || (input.get(1)? > &0.5)) as u8 as f64),
        "not" => Some((!(input.get(0)? > &0.5)) as u8 as f64),
        _ => None,
    }
}

fn default_update_interval(epochs: usize) -> usize {
    if epochs < 100_000 {
        std::cmp::max(1, epochs / 100)
    } else {
        1_000
    }
}
