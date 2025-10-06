#![allow(dead_code)]

//! Neural network core implemented with `ndarray` and friends for a polished developer experience.

pub mod activation;
pub mod layer;
pub mod network;
pub mod optimizer;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "profiling")]
pub mod profiling;

#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {{
        #[cfg(feature = "profiling")]
        let _profile_guard = $crate::profiling::ProfileGuard::new($name);
    }};
}
