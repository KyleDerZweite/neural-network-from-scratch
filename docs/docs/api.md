---
sidebar_position: 5
---

# Web API

The `web-interface` crate provides a RESTful API for interacting with neural networks programmatically.

## Starting the Server

```bash
cargo run --bin web-interface
```

The server runs on `http://localhost:8080` by default.

## API Endpoints

### POST /api/train
Train a neural network on a specified task.

**Request Body:**
```json
{
  "task": "xor" | "sin",
  "learning_rate": 0.1,
  "epochs": 10000,
  "hidden_neurons": 32
}
```

**Response:**
```json
{
  "status": "success",
  "losses": [0.5, 0.3, 0.1, ...],
  "final_loss": 0.001,
  "training_time_ms": 1500
}
```

### POST /api/predict
Make predictions using a trained network.

**Request Body:**
```json
{
  "task": "xor",
  "inputs": [[0, 0], [0, 1], [1, 0], [1, 1]]
}
```

**Response:**
```json
{
  "status": "success",
  "predictions": [0.01, 0.99, 0.98, 0.02]
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z"
}
```

## Web Interface

The web interface provides a graphical way to interact with the neural network:

- **Training Interface**: Configure and run training tasks
- **Visualization**: View training loss curves and network predictions
- **Interactive Demo**: Test the network with custom inputs

Access the web interface at `http://localhost:8080` after starting the server.

## Implementation Details

### Server Setup
```rust
use warp::Filter;

#[tokio::main]
async fn main() {
    let train_route = warp::path!("api" / "train")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(handle_train);

    let routes = train_route
        .or(predict_route)
        .or(health_route)
        .or(static_files);

    warp::serve(routes)
        .run(([127, 0, 0, 1], 8080))
        .await;
}
```

### Request Handling
Each endpoint deserializes the JSON request, performs the computation using `nn-core`, and returns a JSON response.

### Static File Serving
The web interface serves static HTML, CSS, and JavaScript files from the `static/` directory.

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server-side errors

Error responses include a JSON object with an error message:
```json
{
  "status": "error",
  "message": "Invalid task parameter"
}
```

## CORS Support

The API includes CORS headers to allow cross-origin requests from web applications.

## Performance

- **Asynchronous Processing**: Uses Tokio for non-blocking I/O
- **Efficient Computation**: Leverages the optimized `nn-core` library
- **Memory Management**: Proper cleanup of training data