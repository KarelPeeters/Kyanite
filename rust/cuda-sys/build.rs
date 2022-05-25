extern crate core;

use std::fmt::Debug;
use std::path::PathBuf;

use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};
use bindgen::{Builder, CargoCallbacks, EnumVariation};

#[cfg(target_family = "windows")]
fn get_var_path(name: &str) -> PathBuf {
    println!("rerun-if-env-changed={}", name);

    use std::env::VarError;
    let path = std::env::var(name).unwrap_or_else(|e| match e {
        VarError::NotPresent => panic!("Environment variable {} is not defined", name),
        VarError::NotUnicode(_) => panic!("Environment variable {} contains non-unicode path", name),
    });

    println!("Using {}={:?}", name, path);

    let path = PathBuf::from(path);
    if !path.exists() {
        panic!("Path {}={:?} does not exist", name, path);
    }

    path
}

#[cfg(target_family = "windows")]
fn link_cuda() -> Vec<PathBuf> {
    let cuda_path = get_var_path("CUDA_PATH");
    let cudnn_path = get_var_path("CUDNN_PATH");

    for path in [&cuda_path, &cudnn_path] {
        println!(
            "cargo:rustc-link-search=native={}",
            path.join("lib/x64").to_str().unwrap()
        );
    }

    vec![
        cuda_path.join("include"),
        cuda_path.join("include/nvtx3"),
        cudnn_path.join("include"),
    ]
}

#[cfg(target_family = "unix")]
fn link_cuda() -> Vec<PathBuf> {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    vec![
        PathBuf::from("/usr/local/cuda/include/"),
        PathBuf::from("/usr/local/cuda/include/nvtx3"),
    ]
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
    let out_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let builder = Builder::default();

    // tell cargo to link cuda libs
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=nvrtc");

    // find include directories and set up link search paths
    let include_paths = link_cuda();
    let builder = include_paths.iter().fold(builder, |builder, path| {
        let path = path.to_str().unwrap();

        println!("cargo:rerun-if-changed={}", path);
        builder.clang_arg(format!("-I{}", path))
    });

    println!("cargo:rerun-if-changed=wrapper.h");

    builder
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
