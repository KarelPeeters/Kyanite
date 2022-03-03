#![warn(missing_debug_implementations)]

pub use cuda_sys::wrapper::handle::Device;

pub mod executor;
pub mod tester;

mod planner;
mod shape;
mod tensor;

//TODO make this private again?
pub mod kernels;
