/// Game specific code.

pub mod ataxx_utils;
pub mod ataxx_output;
pub mod ataxx_cnn_network;
#[cfg(feature = "tch")]
pub mod ataxx_torch_network;