/// This crate tries to eliminate the global "current device" cuda state.
/// Every cuda/cudnn call that depends on the device is preceded by a cudaSetDevice call.

pub mod status;
pub mod handle;

pub mod event;
pub mod mem;

pub mod descriptor;
pub mod operation;

pub mod group;
