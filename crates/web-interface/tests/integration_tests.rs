use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::{get, post},
    Router,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;
use tower_http::services::ServeDir;
use web_interface::api;

async fn app() -> Router {
    let app_state = Arc::new(api::AppState {
        training_jobs: Mutex::new(HashMap::new()),
        trained_networks: Mutex::new(HashMap::new()),
    });

    Router::new()
        .route("/train", post(api::train))
        .route("/training-progress/:id", get(api::get_training_progress))
        .route("/predict", post(api::predict))
        .route("/stop-training/:id", post(api::stop_training))
        .fallback_service(ServeDir::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/static"
        )))
        .with_state(app_state)
}

#[tokio::test]
async fn test_train_mathematical_function() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/train")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{
                        "network_config": {
                            "layers": [1, 32, 1],
                            "activation_function": "sigmoid",
                            "learning_rate": 0.01,
                            "epochs": 100
                        },
                        "target_type": "function",
                        "target_expression": "sin(x)"
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_train_logical_operation_xor() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/train")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{
                        "network_config": {
                            "layers": [2, 2, 1],
                            "activation_function": "sigmoid",
                            "learning_rate": 0.1,
                            "epochs": 100
                        },
                        "target_type": "logical_operation",
                        "target_expression": "xor"
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
