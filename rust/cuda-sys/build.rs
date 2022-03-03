use std::env;
use std::fmt::Debug;
use std::path::PathBuf;

use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};
use bindgen::{Builder, CargoCallbacks, EnumVariation};

//TODO rewrite this thing again to find cuda automatically (env Var & default location),
// and verify that cudnn is installed

#[cfg(target_family = "windows")]
fn link_cuda(builder: Builder) -> Builder {
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3\\lib\\x64\\");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cublas");

    builder
        .clang_arg("-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
        .clang_arg("-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include/nvtx3")
}

#[cfg(target_family = "unix")]
fn link_cuda(builder: Builder) -> Builder {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cublas");

    builder
        .clang_arg("-I/usr/local/cuda/include/")
        .clang_arg("-I/usr/local/cuda/include/nvtx3")
}

// see https://github.com/rust-lang/rust-bindgen/issues/687,
// specifically https://github.com/rust-lang/rust-bindgen/issues/687#issuecomment-450750547
const IGNORED_MACROS: &[&str] = &[
    "FP_INFINITE",
    "FP_NAN",
    "FP_NORMAL",
    "FP_SUBNORMAL",
    "FP_ZERO",
    "IPPORT_RESERVED",
];

#[derive(Debug)]
struct CustomParseCallBacks;

impl ParseCallbacks for CustomParseCallBacks {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if IGNORED_MACROS.contains(&name) {
            MacroParsingBehavior::Ignore
        } else {
            MacroParsingBehavior::Default
        }
    }

    // redirect to normal handler
    fn include_file(&self, filename: &str) {
        CargoCallbacks.include_file(filename)
    }
}

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    let out_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    link_cuda(Builder::default())
        // input
        .header("wrapper.h")
        .parse_callbacks(Box::new(CustomParseCallBacks))
        // settings
        .size_t_is_usize(true)
        //TODO correctly handle this non-exhaustiveness in FFI
        .default_enum_style(EnumVariation::Rust { non_exhaustive: true })
        .must_use_type("cudaError")
        .must_use_type("cudnnStatus_t")
        .must_use_type("cublasStatus_t")
        .layout_tests(false)
        // output
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
