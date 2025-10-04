#![allow(dead_code)]

//! Core neural network primitives and training logic.

pub mod activation;
pub mod layer;
pub mod matrix;
pub mod network;

#[cfg(feature = "profiling")]
pub mod profiling;

#[macro_export]
macro_rules! profile_scope {
	($name:expr) => {{
		#[cfg(feature = "profiling")]
		let _profile_guard = $crate::profiling::ProfileGuard::new($name);
	}};
}
