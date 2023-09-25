#![warn(missing_debug_implementations)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]

//! A -sys crate for the following cuda libraries and headers:
//! * Cuda (`cuda.h` and `cuda_runtime.h`)
//! * cuDNN (`cudnn.h`)
//! * cuBLAS (`cublas_v2.h` and `cublasLt.h`)
//! * NVRTC (`nvrtc.h`)
//!
//! The [bindings] module contains the FFI [bindgen](https://crates.io/crates/bindgen)-generated signatures.
//! The [wrapper]  module contains more user-friendly and where possible safe wrappers around certain objects.
//!
//! This crate is part of the [Kyanite](https://github.com/KarelPeeters/Kyanite) project, see its readme for more information.

pub mod bindings;
pub mod wrapper;
