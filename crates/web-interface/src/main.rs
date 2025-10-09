use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::services::ServeDir;

use web_interface::{
    api::{get_training_progress, predict, stop_training, train, get_network_internals, forward_pass},
    AppState,
};

#[tokio::main]
async fn main() {
    let app_state = Arc::new(AppState {
        training_jobs: Mutex::new(HashMap::new()),
        trained_networks: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/train", post(train))
        .route("/training-progress/:id", get(get_training_progress))
        .route("/predict", post(predict))
        .route("/stop-training/:id", post(stop_training))
        .route("/network-internals/:id", get(get_network_internals))
        .route("/forward-pass", post(forward_pass))
        .with_state(Arc::clone(&app_state))
        .fallback_service(ServeDir::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/static"
        )));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("failed to bind web interface port");
    axum::serve(listener, app)
        .await
        .expect("web interface server encountered an error");
}
