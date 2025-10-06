---
title: AI-Assisted Variant (`nn-core-library`)
sidebar_position: 5
description: Overview of the AI-generated nn-core-library crate and how it compares to nn-core.
---

# AI-Assisted Variant (`nn-core-library`)

Beyond the hand-written `nn-core` crate, this repository contains `nn-core-library`: an expanded implementation generated with help from an automated coding assistant. This document summarises its capabilities and contrasts them with the canonical code path.

## Purpose

`nn-core-library` serves as a showcase of what AI tooling can scaffold rapidly:

- Richer feature surface, including GPU acceleration for large dense operations.
- Configurable optimisers (SGD and Adam) with persistent state across layers.
- Modular design aimed at production-hardening rather than teaching simplicity.

Treat it as an "advanced preview" rather than the baseline curriculum.

## Key Modules

- [`gpu.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core-library/src/gpu.rs) — wraps `wgpu` to execute matrix multiplications on compatible hardware. Workloads automatically fall back to the CPU if the GPU path errors or the problem size is too small.
- [`optimizer.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core-library/src/optimizer.rs) — provides both vanilla SGD and Adam with bias correction terms, mirroring contemporary deep-learning toolkits.
- [`network.rs`](https://github.com/KyleDerZweite/neural-network-from-scratch/blob/master/crates/nn-core-library/src/network.rs) — integrates the GPU helper and optimiser state while keeping APIs similar to `nn-core`.

## Design Trade-offs

| Aspect | `nn-core` | `nn-core-library` |
| --- | --- | --- |
| Target audience | Learning & research | Performance exploration |
| Matrix backend | Hand-rolled `Matrix` type | `ndarray` with optional GPU offload |
| Optimisation | SGD only | SGD + Adam |
| Hardware support | CPU | CPU + optional GPU (`wgpu`) |
| Complexity | Minimal | Significantly higher |

## When to Use Which

- **Stick with `nn-core`** when you need transparency, deterministic behaviour, or a controlled environment for coursework.
- **Experiment with `nn-core-library`** when you want to validate GPU speed-ups, try advanced optimisers, or inspect alternative architectural decisions produced by AI assistance.

Both crates share the same mathematical foundations. The difference lies in engineering scope and ergonomics.
