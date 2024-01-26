//! The NVRTC wrapper.
//!
//! The most important types are:
//! * [CuModule](self::rtc::core::CuModule): a compiled cuda module
//! * [CuFunction](self::rtc::core::CuFunction) a compiled cuda kernel
//! * [KernelArgs](self::rtc::args::KernelArgs): a utility to encode kernel arguments as bytes.
//!
//! The typical workflow, compiling a (very inefficient) memcpy kernel:
//! ```
//! # use std::collections::HashMap;
//! # use kn_cuda_sys::wrapper::handle::{CudaStream, CudaDevice};
//! # use kn_cuda_sys::wrapper::rtc::args::KernelArgs;
//! # use kn_cuda_sys::wrapper::rtc::core::{CuModule};
//! # use kn_cuda_sys::wrapper::status::Status;
//! // define the source code
//! let source = r#"
//! typedef unsigned char u8;
//! __global__ void kernel_memcpy(u8* dst, u8* src, int n) {
//!     for (int i = 0; i < n; i++) {
//!       dst[i] = src[i];
//!     }
//! }"#;
//! let kernel_name = "kernel_memcpy";
//!
//! // select a device
//! let device = CudaDevice::new(0);
//! let stream = CudaStream::new(device);
//!
//! // compile the module, indicating which function(s) we want to use later
//! let result = CuModule::from_source(device, source, None, &[kernel_name], &HashMap::new());
//!
//! // print warnings and errors if any
//! if !result.log.is_empty() {
//!     eprintln!("Source:\n{}\nLog:\n{}", result.source_with_line_numbers(), result.log);
//! }
//!
//! // get the kernel function
//! let kernel = result.get_function_by_name(kernel_name).unwrap().unwrap();
//!
//! // allocate inputs and outputs
//! let n: i32 = 16;
//! let ptr_dest = device.alloc(n as usize);
//! let ptr_src = device.alloc(n as usize);
//!
//! unsafe {
//!     // build kernel args
//!     let mut args = KernelArgs::new();
//!     args.push(ptr_dest.ptr());
//!     args.push(ptr_src.ptr());
//!     args.push_int(n);
//!     let args = args.finish();
//!
//!     // actually launch the kernel
//!     kernel.launch_kernel(1, 1, 0, &stream, &args).unwrap();
//! }
//!
//! // wait for the kernel to complete
//! stream.synchronize();
//! ```
//!
//! Modules and functions are reference counted to enable automatic memory management.

/// Kernel argument builder.
pub mod args;
/// Core abstractions and utilities.
pub mod core;
