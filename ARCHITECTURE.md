# Architecture: Modular Rust Workspace for From-Scratch Neural Network and Reference Comparison

This document describes the modular architecture for a Rust-based, from-scratch feed-forward neural network (MLP) plus a reference/"official" adapter layer to validate results side-by-side. The goal is to keep core learning logic and linear algebra self-contained (no ML frameworks) while making it easy to run the same experiments using a well-maintained Rust ML stack for verification.


## Goals and Design Principles

- Separation of concerns
  - Scratch Linear Algebra: from-scratch matrices/vectors and ops.
  - Scratch NN: training and inference for MLP using the scratch linalg.
  - Datasets/Preprocessing: data generation/normalization, reusable across backends.
  - Metrics: loss/accuracy utilities for consistent evaluation.
  - Reference Adapters: thin wrappers using established libraries, feature-gated and optional.
- Reproducibility: Same dataset generators, metrics, and evaluation harness for both scratch and reference paths.
- Build stability: The default build uses only the scratch crates (no external heavy dependencies). Reference adapters are behind features.
- Extensibility: Easy to add other backends (e.g., different linalg or different NN libraries) or new tasks.


## Workspace Layout

- apps and libs are split into small crates to keep compile times reasonable and responsibilities clear.
- Default members are the scratch path; reference crates are included but have optional deps and compile without activating features.

```
neuralnet/               (workspace root, also a small CLI app for now)
├─ ARCHITECTURE.md       (this document)
├─ README.md             (course-level readme; to be updated to Rust path later)
├─ Cargo.toml            (workspace + root binary crate)
├─ src/
│  └─ main.rs            (temporary CLI placeholder)
└─ crates/
   ├─ linalg_scratch/    (from-scratch vector/matrix ops)
   │  ├─ Cargo.toml
   │  └─ src/lib.rs
   ├─ nn_scratch/        (MLP definitions + training using linalg_scratch)
   │  ├─ Cargo.toml
   │  └─ src/lib.rs
   ├─ datasets/          (XOR, sine sampler, normalization helpers)
   │  ├─ Cargo.toml
   │  └─ src/lib.rs
   ├─ metrics/           (MSE, accuracy, confusion, etc.)
   │  ├─ Cargo.toml
   │  └─ src/lib.rs
   └─ nn_reference/      (feature-gated adapters for official libs)
      ├─ Cargo.toml
      └─ src/lib.rs
```


## Crates and Responsibilities

- linalg_scratch
  - Implements basic vector/matrix types and operations.
  - No external numeric dependencies to honor the "from-scratch" requirement.
  - Provides deterministic APIs for reproducibility and testability.
- nn_scratch
  - Implements feed-forward MLP, forward/backward passes, and SGD training using linalg_scratch.
  - Exposes a clean trait-based interface to make swapping backends feasible later.
- datasets
  - Provides XOR dataset, sine sampler (with configurable interval and density), and normalization utilities.
  - Ensures identical data for scratch and reference comparisons.
- metrics
  - Provides MSE, accuracy, and simple helpers to summarize results.
  - Stays decoupled from any tensor type; accepts plain slices/Vecs where feasible.
- nn_reference
  - Feature-gated adapters to well-known, actively maintained Rust libraries.
  - Goals: validate numerical outputs and learning behavior, not to depend on them at runtime for the course project.


## Reference Libraries and Integration Strategy

Reference backends are optional and compile only when their features are enabled. This ensures the default build stays lightweight and free of heavy dependencies.

- burn (feature: `burn`)
  - Burn is an actively maintained deep learning framework in Rust.
  - Provides modern training loops, autograd, modules, and backends (e.g., CPU, wgpu).
  - We will implement a small MLP module and use the same datasets/metrics to compare with our scratch implementation.
- ndarray (feature: `ndarray`)
  - ndarray is a widely used N-dimensional array library in Rust.
  - We can implement a lightweight reference MLP using ndarray operations to compare against our scratch path while still staying in pure Rust.
  - This is useful as a middle ground if full DL frameworks are too heavy.

Both adapters will:
- Use identical data preprocessing and evaluation.
- Provide functions like `train_xor_ref`, `train_sin_ref`, returning predictions and metrics that mirror the scratch path.


## Data Flow and Contracts

Common contracts for both backends:
- Inputs: contiguous numeric slices or Vec<Vec<f32>> (for simplicity) with dimensions documented per function.
- Targets: same shape as outputs; classification can be one-hot or binary.
- Outputs: predictions with shapes matching targets; metrics consume these.
- Errors: return Result<T, E> with clear error types; avoid panics in library code.


## Testing and Verification Plan

- Unit tests in linalg_scratch for core ops (mul, add, transpose, hadamard), including shape errors.
- Unit tests in nn_scratch for forward pass with fixed weights, and backprop gradient sanity checks (finite differences on tiny nets).
- Golden tests in datasets and metrics to pin expected values.
- Cross-backend parity tests (when a reference feature is enabled):
  - XOR: After training N epochs with the same seed and learning rate, compare predictions and MSE within a small tolerance.
  - Sine: Evaluate on a fixed grid and compare MSE curves.


## Build Modes

- Default (scratch-only):
  - No external ML dependencies.
  - Suitable for the course submission and exams.
- With reference (opt-in):
  - Enable `nn_reference` features to compile adapters.
  - Example: `--features nn_reference/burn` or `--features nn_reference/ndarray` from the crate directory.


## Future CLI Structure (Placeholder)

The root binary will evolve into a small CLI front-end:
- `neuralnet --task xor --backend scratch`
- `neuralnet --task xor --backend burn` (requires feature)
- `neuralnet --task sin --backend scratch --epochs 20000 --lr 0.01`

For now, the CLI is a stub; implementation will follow after the library crates are fleshed out.


## Rationale and Brief Research Notes

- Rust Workspace Modularity: Recommended for medium-sized Rust projects to isolate concerns, reduce compile times, and support feature-gated optional integrations. References: Rust Book (Workspaces), community templates (e.g., cargo workspaces used by large Rust projects like Tokio, Bevy).
- burn: Modern Rust DL framework with active development, modular backends, and a clear API for MLPs. Using it as an opt-in reference validates the math/path without relying on it for the core project.
- ndarray: De facto standard for numerical arrays in Rust; suitable for a minimal reference MLP if a full DL stack is unnecessary.

This design gives you a clear, didactic path for the from-scratch build while enabling rigorous comparisons against established libraries.

