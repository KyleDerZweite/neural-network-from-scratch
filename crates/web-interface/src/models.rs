use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NeuralNetworkConfiguration {
    pub layers: Vec<usize>,
    pub activation_function: String,
    pub learning_rate: f64,
    pub epochs: usize,
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
