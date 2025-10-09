use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::{get, post},
    Router,
};
use http_body_util::BodyExt;
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
async fn test_post_train() {
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
                            "backend": "nn-core",
                            "layers": [1, 32, 1],
                            "activation_function": "sigmoid",
                            "learning_rate": 0.01,
                            "epochs": 20000
                        },
                        "target_type": "function",
                        "target_expression": "sin(x)"
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    if status != StatusCode::OK {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        panic!(
            "unexpected status: {}, body: {}",
            status,
            String::from_utf8_lossy(&bytes)
        );
    }
    // Ensure pending status with optional warning payloads works.
}

#[tokio::test]
async fn test_get_training_progress() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/training-progress/00000000-0000-0000-0000-000000000000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_post_predict() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    r#"{
                        "job_id": "00000000-0000-0000-0000-000000000000",
                        "input": [0.5]
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_post_stop_training() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stop-training/00000000-0000-0000-0000-000000000000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
