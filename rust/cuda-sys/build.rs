use std::env;
use std::path::PathBuf;

use bindgen::EnumVariation;

fn main() {
    println!("cargo:rustc-link-lib=\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3\\lib\\x64\\cuda");
    println!("cargo:rustc-link-lib=\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3\\lib\\x64\\cudart");
    println!("cargo:rustc-link-lib=\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3\\lib\\x64\\cudnn");

    println!("cargo:rerun-if-changed=wrapper.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindgen::Builder::default()
        // input
        .clang_arg("-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
        .clang_arg("-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include/nvtx3")
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))

        // settings
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
        .must_use_type("cudaError")
        .must_use_type("cudnnStatus_t")
        .must_use_type("cublasStatus_t")

        // output
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
