use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::{get, post},
    Router,
};
use tower::ServiceExt; // for `app.oneshot()`
use tower_http::services::ServeDir;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

// Import the API and models from the main crate
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
        .fallback_service(ServeDir::new(concat!(env!("CARGO_MANIFEST_DIR"), "/crates/web-interface/static")))
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
                            "layers": [2, 32, 1],
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

    assert_eq!(response.status(), StatusCode::OK); // Expecting 200 OK for now
    let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    // assert_eq!(&body[..], b"Hello, World!"); // Placeholder assertion
}

#[tokio::test]
async fn test_get_training_progress() {
    let app = app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/training-progress/some-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND); // Expecting 404 Not Found for now
    // let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    // assert_eq!(&body[..], b"Progress"); // Placeholder assertion
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

    assert_eq!(response.status(), StatusCode::NOT_FOUND); // Expecting 404 Not Found for now
    // let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    // assert_eq!(&body[..], b"Prediction"); // Placeholder assertion
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

    assert_eq!(response.status(), StatusCode::NOT_FOUND); // Expecting 404 Not Found for now
    // let body = hyper::body::to_bytes(response.into_body()).await.unwrap();
    // assert_eq!(&body[..], b"Stopped"); // Placeholder assertion
}