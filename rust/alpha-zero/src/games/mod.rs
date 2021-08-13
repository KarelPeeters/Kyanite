/// Game specific code.

pub mod ataxx_utils;
pub mod ataxx_output;
pub mod ataxx_cnn_network;
pub mod ataxx_cpu_network;
#[cfg(feature = "tch")]
pub mod ataxx_torch_network;
#[cfg(feature="onnxruntime")]
pub mod ataxx_onnx_network;