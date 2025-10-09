use std::cmp::min;
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
};
use meval::{Error as MevalError, Expr};
use nn_core::{layer::Layer as CoreLayer, network::NeuralNetwork as CoreNetwork};
use nn_core_library::{
    layer::Layer as LibraryLayer, network::NeuralNetwork as LibraryNetwork,
    optimizer::OptimizerKind as LibraryOptimizerKind,
};
use tokio::task;
use uuid::Uuid;

use crate::models::{
    BackendKind, CoreLibraryOptions, PredictRequest, PredictResponse, SamplePrediction,
    StopTrainingResponse, TrainRequest, TrainResponse, TrainingJob, TrainingProgressResponse,
    TrainingStatus, NetworkInternalsResponse, LayerInternals, WeightStats,
    ForwardPassRequest, ForwardPassResponse,
};

pub struct AppState {
    pub training_jobs: Mutex<HashMap<Uuid, TrainingJob>>,
    pub trained_networks: Mutex<HashMap<Uuid, TrainedModel>>,
}

pub enum TrainedModel {
    Core(CoreNetwork),
    CoreLibrary(LibraryNetwork),
}

const MAX_PREVIEW_SAMPLES: usize = 50;

trait PredictiveNetwork {
    fn predict_values(&self, input: &Vec<f64>) -> Vec<f64>;
}

impl PredictiveNetwork for CoreNetwork {
    fn predict_values(&self, input: &Vec<f64>) -> Vec<f64> {
        self.predict(input)
    }
}

impl PredictiveNetwork for LibraryNetwork {
    fn predict_values(&self, input: &Vec<f64>) -> Vec<f64> {
        self.predict(input.as_slice())
    }
}

trait TrainableNetwork: PredictiveNetwork {
    fn train_epochs(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize);
}

impl TrainableNetwork for CoreNetwork {
    fn train_epochs(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize) {
        self.train(inputs, targets, epochs);
    }
}

impl TrainableNetwork for LibraryNetwork {
    fn train_epochs(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize) {
        self.train(inputs.as_slice(), targets.as_slice(), epochs);
    }
}

impl TrainedModel {
    fn predict(&self, input: &[f64]) -> Vec<f64> {
        match self {
            TrainedModel::Core(network) => network.predict(&input.to_vec()),
            TrainedModel::CoreLibrary(network) => network.predict(input),
        }
    }
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
    let mut warning = match validate_network_config(&payload.network_config) {
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

    if let BackendKind::CoreLibrary = payload.network_config.backend {
        if let Some(options) = &payload.network_config.core_library_options {
            if options.use_gpu.unwrap_or(false) {
                #[cfg(not(feature = "gpu"))]
                {
                    let message = "GPU acceleration requested but server was built without the nn-core-library/gpu feature";
                    warning = Some(match warning {
                        Some(existing) if !existing.is_empty() => format!("{existing}. {message}"),
                        _ => message.to_string(),
                    });
                }
            }
        }
    }

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
        prediction_preview: Vec::new(),
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
                backend: job.network_config.backend.clone(),
                target_type: job.target_type.clone(),
                prediction_preview: job.prediction_preview.clone(),
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
                backend: BackendKind::Core,
                target_type: "unknown".into(),
                prediction_preview: Vec::new(),
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

    if let Some(model) = networks.get(&payload.job_id) {
        let prediction = model.predict(&payload.input);

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

    let trained_model = match job_snapshot.network_config.backend {
        BackendKind::Core => {
            let layers = match build_core_layers(
                &job_snapshot.network_config.layers,
                &job_snapshot.network_config.activation_function,
            ) {
                Ok(layers) => layers,
                Err(error) => {
                    update_job_status(&app_state, job_id, TrainingStatus::Failed, Some(error));
                    return;
                }
            };

            let mut network = CoreNetwork::new(layers, job_snapshot.network_config.learning_rate);
            run_training_loop(
                &mut network,
                &inputs,
                &targets,
                job_snapshot.network_config.epochs,
                update_interval,
                job_id,
                &app_state,
            );
            Some(TrainedModel::Core(network))
        }
        BackendKind::CoreLibrary => {
            let layers = match build_library_layers(
                &job_snapshot.network_config.layers,
                &job_snapshot.network_config.activation_function,
            ) {
                Ok(layers) => layers,
                Err(error) => {
                    update_job_status(&app_state, job_id, TrainingStatus::Failed, Some(error));
                    return;
                }
            };

            let optimizer = job_snapshot
                .network_config
                .core_library_options
                .as_ref()
                .and_then(|options| options.optimizer.as_deref())
                .map(optimizer_from_str)
                .unwrap_or_else(Default::default);

            let mut network = LibraryNetwork::with_optimizer(
                layers,
                job_snapshot.network_config.learning_rate,
                optimizer,
            );

            if let Some(options) = &job_snapshot.network_config.core_library_options {
                configure_library_backend(&mut network, options);
            }

            run_training_loop(
                &mut network,
                &inputs,
                &targets,
                job_snapshot.network_config.epochs,
                update_interval,
                job_id,
                &app_state,
            );
            Some(TrainedModel::CoreLibrary(network))
        }
    };

    if let Some(model) = trained_model {
        finalize_training(job_id, &app_state);
        app_state
            .trained_networks
            .lock()
            .expect("trained networks mutex poisoned")
            .insert(job_id, model);
    }
}

fn run_training_loop<N: TrainableNetwork + ?Sized>(
    network: &mut N,
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<Vec<f64>>,
    total_epochs: usize,
    update_interval: usize,
    job_id: Uuid,
    app_state: &Arc<AppState>,
) {
    if total_epochs == 0 {
        return;
    }

    let mut completed_epochs = 0usize;
    let step = update_interval.max(1);

    while completed_epochs < total_epochs {
        let remaining = total_epochs - completed_epochs;
        let chunk = min(step, remaining);
        network.train_epochs(inputs, targets, chunk);
        completed_epochs += chunk;

        update_training_progress_generic(
            network,
            inputs,
            targets,
            total_epochs,
            completed_epochs,
            job_id,
            app_state,
        );

        if is_job_stopped(app_state, job_id) {
            break;
        }
    }
}

fn update_training_progress_generic<N: PredictiveNetwork + ?Sized>(
    network: &N,
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<Vec<f64>>,
    total_epochs: usize,
    completed_epochs: usize,
    job_id: Uuid,
    app_state: &Arc<AppState>,
) {
    if inputs.is_empty() {
        return;
    }

    let mut mse = 0.0;
    let mut total = 0usize;
    for (input_vec, target_vec) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict_values(input_vec);
        for (pred, target) in prediction.iter().zip(target_vec.iter()) {
            mse += (pred - target).powi(2);
            total += 1;
        }
    }
    if total > 0 {
        mse /= total as f64;
    }

    let preview = generate_prediction_preview(network, inputs, targets);
    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");

    if let Some(job) = jobs.get_mut(&job_id) {
        let total_epochs = total_epochs.max(1);
        job.progress = (completed_epochs as f64 / total_epochs as f64) * 100.0;
        job.current_loss = Some(mse);
        job.loss_history.push(mse);
        job.prediction_preview = preview;
    }
}

fn generate_prediction_preview<N: PredictiveNetwork + ?Sized>(
    network: &N,
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<Vec<f64>>,
) -> Vec<SamplePrediction> {
    if inputs.is_empty() || targets.is_empty() {
        return Vec::new();
    }

    let step = ((inputs.len() + MAX_PREVIEW_SAMPLES - 1) / MAX_PREVIEW_SAMPLES).max(1);
    let mut preview = Vec::new();

    for idx in (0..inputs.len()).step_by(step) {
        if let Some(target) = targets.get(idx) {
            let input = inputs[idx].clone();
            let prediction = network.predict_values(&inputs[idx]);
            preview.push(SamplePrediction {
                input,
                prediction,
                target: target.clone(),
            });
            if preview.len() >= MAX_PREVIEW_SAMPLES {
                break;
            }
        }
    }

    preview
}

fn finalize_training(job_id: Uuid, app_state: &Arc<AppState>) {
    let mut jobs = app_state
        .training_jobs
        .lock()
        .expect("training jobs mutex poisoned");
    if let Some(job) = jobs.get_mut(&job_id) {
        if !matches!(job.status, TrainingStatus::Stopped | TrainingStatus::Failed) {
            job.status = TrainingStatus::Completed;
        }
    }
}

fn optimizer_from_str(name: &str) -> LibraryOptimizerKind {
    match name.to_lowercase().as_str() {
        "sgd" => LibraryOptimizerKind::sgd(),
        "rmsprop" => LibraryOptimizerKind::rmsprop(),
        _ => LibraryOptimizerKind::adam(),
    }
}

fn configure_library_backend(network: &mut LibraryNetwork, options: &CoreLibraryOptions) {
    #[cfg(feature = "gpu")]
    {
        let use_gpu = options.use_gpu.unwrap_or(false);
        if use_gpu {
            let threshold = options.gpu_workload_threshold.unwrap_or(1);
            network.set_gpu_workload_threshold(threshold);
        } else {
            network.set_gpu_workload_threshold(usize::MAX);
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = network;
        let _ = options;
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

    if let BackendKind::CoreLibrary = config.backend {
        if let Some(options) = &config.core_library_options {
            if let Some(threshold) = options.gpu_workload_threshold {
                if threshold == 0 {
                    return Err("GPU workload threshold must be greater than zero".into());
                }
            }
            if let Some(optimizer) = &options.optimizer {
                let lower = optimizer.to_lowercase();
                if !matches!(lower.as_str(), "adam" | "sgd" | "rmsprop") {
                    return Err("Unsupported optimizer for nn-core-library".into());
                }
            }
        }
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
            let evaluator = expr.bind("x").map_err(|err: MevalError| err.to_string())?;
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

fn build_core_layers(
    layer_sizes: &[usize],
    activation_function: &str,
) -> Result<Vec<CoreLayer>, String> {
    if layer_sizes.len() < 2 {
        return Err("At least two layer sizes are required".into());
    }
    let mut layers = Vec::new();
    for index in 0..layer_sizes.len() - 1 {
        let input_size = layer_sizes[index];
        let output_size = layer_sizes[index + 1];
        // Use the specified activation function for all layers including output
        // This allows sigmoid on output for classification tasks like XOR
        let activation = activation_function;
        layers.push(CoreLayer::new(input_size, output_size, activation));
    }
    Ok(layers)
}

fn build_library_layers(
    layer_sizes: &[usize],
    activation_function: &str,
) -> Result<Vec<LibraryLayer>, String> {
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
        layers.push(LibraryLayer::new(input_size, output_size, activation));
    }
    Ok(layers)
}

fn generate_function_data(expression: &str, num_samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..num_samples {
        let x = (i as f64 / num_samples as f64) * 7.0;
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

pub async fn get_network_internals(
    State(app_state): State<Arc<AppState>>,
    Path(job_id): Path<Uuid>,
) -> (StatusCode, Json<NetworkInternalsResponse>) {
    let networks = app_state
        .trained_networks
        .lock()
        .expect("trained networks mutex poisoned");

    if let Some(model) = networks.get(&job_id) {
        let layers = match model {
            TrainedModel::Core(network) => extract_core_layers(network),
            TrainedModel::CoreLibrary(network) => extract_library_layers(network),
        };

        (
            StatusCode::OK,
            Json(NetworkInternalsResponse {
                job_id,
                layers,
                message: None,
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(NetworkInternalsResponse {
                job_id,
                layers: Vec::new(),
                message: Some("Trained network not found".into()),
            }),
        )
    }
}

fn extract_core_layers(network: &CoreNetwork) -> Vec<LayerInternals> {
    network
        .get_layers()
        .iter()
        .enumerate()
        .map(|(idx, layer)| {
            let weights = layer.get_weights();
            let biases = layer.get_biases();
            let weight_stats = calculate_weight_stats(&weights);
            
            LayerInternals {
                layer_index: idx,
                weights,
                biases,
                weight_stats,
            }
        })
        .collect()
}

fn extract_library_layers(network: &LibraryNetwork) -> Vec<LayerInternals> {
    network
        .get_layers()
        .iter()
        .enumerate()
        .map(|(idx, layer)| {
            let weights = layer.get_weights();
            let biases = layer.get_biases();
            let weight_stats = calculate_weight_stats(&weights);
            
            LayerInternals {
                layer_index: idx,
                weights,
                biases,
                weight_stats,
            }
        })
        .collect()
}

fn calculate_weight_stats(weights: &Vec<Vec<f64>>) -> WeightStats {
    let flat_weights: Vec<f64> = weights.iter().flat_map(|row| row.iter().copied()).collect();
    
    if flat_weights.is_empty() {
        return WeightStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
        };
    }
    
    let min = flat_weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max = flat_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = flat_weights.iter().sum::<f64>() / flat_weights.len() as f64;
    
    let variance = flat_weights
        .iter()
        .map(|&w| (w - mean).powi(2))
        .sum::<f64>()
        / flat_weights.len() as f64;
    let std_dev = variance.sqrt();
    
    WeightStats {
        min,
        max,
        mean,
        std_dev,
    }
}

pub async fn forward_pass(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<ForwardPassRequest>,
) -> (StatusCode, Json<ForwardPassResponse>) {
    let networks = app_state
        .trained_networks
        .lock()
        .expect("trained networks mutex poisoned");

    if let Some(model) = networks.get(&payload.job_id) {
        let (activations, weighted_sums, output) = match model {
            TrainedModel::Core(network) => perform_core_forward_pass(network, &payload.input),
            TrainedModel::CoreLibrary(network) => perform_library_forward_pass(network, &payload.input),
        };

        (
            StatusCode::OK,
            Json(ForwardPassResponse {
                job_id: payload.job_id,
                input: payload.input,
                activations,
                weighted_sums,
                output,
                message: None,
            }),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ForwardPassResponse {
                job_id: payload.job_id,
                input: payload.input,
                activations: Vec::new(),
                weighted_sums: Vec::new(),
                output: Vec::new(),
                message: Some("Trained network not found".into()),
            }),
        )
    }
}

fn perform_core_forward_pass(network: &CoreNetwork, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut activations = vec![input.to_vec()];
    let mut weighted_sums = Vec::new();
    
    for layer in network.get_layers() {
        let last_activation = activations.last().unwrap();
        let (activation, weighted_sum) = layer.forward_with_details(last_activation);
        activations.push(activation);
        weighted_sums.push(weighted_sum);
    }
    
    let output = activations.last().unwrap().clone();
    (activations, weighted_sums, output)
}

fn perform_library_forward_pass(network: &LibraryNetwork, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut activations = vec![input.to_vec()];
    let mut weighted_sums = Vec::new();
    
    for layer in network.get_layers() {
        let last_activation = activations.last().unwrap();
        let (activation, weighted_sum) = layer.forward_with_details(last_activation);
        activations.push(activation);
        weighted_sums.push(weighted_sum);
    }
    
    let output = activations.last().unwrap().clone();
    (activations, weighted_sums, output)
}
