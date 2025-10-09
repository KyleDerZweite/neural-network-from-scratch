use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum BackendKind {
    #[serde(rename = "nn-core")]
    Core,
    #[serde(rename = "nn-core-library")]
    CoreLibrary,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CoreLibraryOptions {
    pub optimizer: Option<String>,
    pub use_gpu: Option<bool>,
    pub gpu_workload_threshold: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NeuralNetworkConfiguration {
    pub backend: BackendKind,
    pub layers: Vec<usize>,
    pub activation_function: String,
    pub learning_rate: f64,
    pub epochs: usize,
    #[serde(default)]
    pub core_library_options: Option<CoreLibraryOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Stopped,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingJob {
    pub id: Uuid,
    pub network_config: NeuralNetworkConfiguration,
    pub target_type: String,
    pub target_expression: String,
    pub status: TrainingStatus,
    pub progress: f64,
    pub current_loss: Option<f64>,
    pub loss_history: Vec<f64>,
    pub update_frequency: Option<usize>,
    pub warning: Option<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub prediction_preview: Vec<SamplePrediction>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainRequest {
    pub network_config: NeuralNetworkConfiguration,
    pub target_type: String,
    pub target_expression: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainResponse {
    pub job_id: Uuid,
    pub status: TrainingStatus,
    pub warning: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingProgressResponse {
    pub job_id: Uuid,
    pub status: TrainingStatus,
    pub progress: f64,
    pub current_loss: Option<f64>,
    pub loss_history: Vec<f64>,
    pub update_frequency: Option<usize>,
    pub warning: Option<String>,
    pub message: Option<String>,
    pub backend: BackendKind,
    pub target_type: String,
    pub prediction_preview: Vec<SamplePrediction>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictRequest {
    pub job_id: Uuid,
    pub input: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    pub job_id: Uuid,
    pub input: Vec<f64>,
    pub prediction: Vec<f64>,
    pub actual_result: Option<f64>,
    pub message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StopTrainingResponse {
    pub job_id: Uuid,
    pub status: TrainingStatus,
    pub message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SamplePrediction {
    pub input: Vec<f64>,
    pub prediction: Vec<f64>,
    pub target: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkInternalsResponse {
    pub job_id: Uuid,
    pub layers: Vec<LayerInternals>,
    pub message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LayerInternals {
    pub layer_index: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub weight_stats: WeightStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeightStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    pub job_id: Uuid,
    pub input: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub job_id: Uuid,
    pub input: Vec<f64>,
    pub activations: Vec<Vec<f64>>,
    pub weighted_sums: Vec<Vec<f64>>,
    pub output: Vec<f64>,
    pub message: Option<String>,
}
