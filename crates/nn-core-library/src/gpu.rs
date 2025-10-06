//! GPU acceleration utilities powered by wgpu.
//!
//! This module provides a thin abstraction over a compute pipeline that
//! accelerates dense matrix multiplications. The rest of the training
//! pipeline can opportunistically use the accelerator when the workload is
//! large enough, while gracefully falling back to CPU computations otherwise.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use wgpu::util::DeviceExt;

/// Workgroup size for the matrix multiplication shader.
const MATMUL_WORKGROUP_SIZE: u32 = 16;

/// Represents the minimum amount of work (measured as `m * n * k`) that should
/// be offloaded to the GPU. Smaller workloads are generally faster on the CPU
/// once memory transfer overhead is considered.
pub const DEFAULT_GPU_WORKLOAD_THRESHOLD: usize = 4_096;

/// A simple GPU compute accelerator that focuses on dense matrix operations.
pub struct GpuAccelerator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bind_group_layout: wgpu::BindGroupLayout,
    adapter_info: wgpu::AdapterInfo,
}

/// Errors that can occur when interacting with the GPU accelerator.
#[derive(Debug)]
pub enum GpuError {
    AdapterNotFound,
    RequestDevice(wgpu::RequestDeviceError),
    BufferCopy(wgpu::BufferAsyncError),
    Wgpu(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::AdapterNotFound => write!(f, "No compatible GPU adapter found"),
            GpuError::RequestDevice(err) => write!(f, "Failed to create GPU device: {err}"),
            GpuError::BufferCopy(err) => write!(f, "Failed to map GPU buffer: {err}"),
            GpuError::Wgpu(msg) => write!(f, "wgpu error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Uniform buffer passed to the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

impl GpuAccelerator {
    /// Attempts to initialise the GPU accelerator. Returns `None` if no GPU is
    /// available or an error occurs during setup.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::AdapterNotFound)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Initialising GPU accelerator on {} {} ({:?})",
            adapter_info.vendor,
            adapter_info.name,
            adapter_info.backend
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nn-core-library GPU device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(GpuError::RequestDevice)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nn-core-library matmul shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        let matmul_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nn-core-library matmul bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nn-core-library matmul pipeline layout"),
            bind_group_layouts: &[&matmul_bind_group_layout],
            push_constant_ranges: &[],
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nn-core-library matmul pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            matmul_pipeline,
            matmul_bind_group_layout,
            adapter_info,
        })
    }

    /// Returns information about the adapter powering this accelerator.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns `true` if the workload should be offloaded to the GPU given the
    /// configured threshold.
    pub fn is_workload_worth_gpu(&self, m: usize, n: usize, k: usize, threshold: usize) -> bool {
        (m * n * k) >= threshold
    }

    /// Performs a dense matrix multiplication using the GPU: `lhs (m×k)` * `rhs (k×n)`.
    pub fn matmul(&self, lhs: &Array2<f64>, rhs: &Array2<f64>) -> Result<Array2<f64>, GpuError> {
        let (m, k_lhs) = lhs.dim();
        let (k_rhs, n) = rhs.dim();
        assert_eq!(k_lhs, k_rhs, "Matrix dimensions are incompatible for matmul");

        let lhs_data: Vec<f32> = lhs.iter().map(|&x| x as f32).collect();
        let rhs_data: Vec<f32> = rhs.iter().map(|&x| x as f32).collect();
        let output_elems = m * n;
        let output_size_bytes = (output_elems * std::mem::size_of::<f32>()) as u64;

        let lhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nn-core-library lhs"),
            contents: bytemuck::cast_slice(&lhs_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rhs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nn-core-library rhs"),
            contents: bytemuck::cast_slice(&rhs_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nn-core-library output"),
            size: output_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nn-core-library staging"),
            size: output_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = MatMulParams {
            m: m as u32,
            n: n as u32,
            k: k_lhs as u32,
            _pad: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nn-core-library params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn-core-library matmul bind group"),
            layout: &self.matmul_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lhs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rhs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nn-core-library matmul encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nn-core-library matmul pass"),
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dispatch_x = (m as u32 + MATMUL_WORKGROUP_SIZE - 1) / MATMUL_WORKGROUP_SIZE;
            let dispatch_y = (n as u32 + MATMUL_WORKGROUP_SIZE - 1) / MATMUL_WORKGROUP_SIZE;
            log::debug!("Dispatching GPU matmul with grid ({dispatch_x}, {dispatch_y}, 1)");
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let map_future = buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(map_future).map_err(GpuError::BufferCopy)?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_f64: Vec<f64> = result.into_iter().map(|x| x as f64).collect();
        Array2::from_shape_vec((m, n), result_f64).map_err(|err| GpuError::Wgpu(err.to_string()))
    }
}

/// WGSL compute shader implementing a naive dense matrix multiplication.
const MATMUL_SHADER: &str = r#"
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let m = params.m;
    let n = params.n;
    let k = params.k;

    if (row >= m || col >= n) {
        return;
    }

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        let lhs_idx = row * k + i;
        let rhs_idx = i * n + col;
        acc = acc + lhs[lhs_idx] * rhs[rhs_idx];
    }

    let out_index = row * n + col;
    out[out_index] = acc;
}
"#;
