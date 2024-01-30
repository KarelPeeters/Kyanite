use std::env::VarError;
use std::fmt::Debug;
use std::path::PathBuf;

use bindgen::{Builder, CargoCallbacks, EnumVariation};
use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};

struct PlatformSpecific {
    fallback_paths: Vec<PathBuf>,
    suffixes_include: Vec<PathBuf>,
    suffixes_link: Vec<PathBuf>,
}

#[cfg(target_family = "windows")]
fn platform() -> PlatformSpecific {
    PlatformSpecific {
        fallback_paths: vec![],
        suffixes_include: vec![PathBuf::from("include"), PathBuf::from("include/nvtx3")],
        suffixes_link: vec![PathBuf::from("lib/x64"), PathBuf::from("lib")],
    }
}

#[cfg(target_family = "unix")]
fn platform() -> PlatformSpecific {
    PlatformSpecific {
        fallback_paths: vec![PathBuf::from("/usr/local/cuda")],
        suffixes_include: vec![PathBuf::from("include"), PathBuf::from("include/nvtx3")],
        suffixes_link: vec![PathBuf::from("lib64"), PathBuf::from("lib")],
    }
}

const CUDA_PATH_VAR: &str = "CUDA_PATH";
const CUDNN_PATH_VAR: &str = "CUDNN_PATH";

const ERROR_MESSAGE: &str = r"
Could not find CUDA installation.
Set CUDA_PATH environment variable to the base directory of your CUDA installation.
CUDNN_PATH can also be optionally set to the base directory of your cuDNN installation.
";

fn error() -> ! {
    panic!("{}", ERROR_MESSAGE.trim());
}

fn find_base_dir(platform: &PlatformSpecific) -> PathBuf {
    match std::env::var(CUDA_PATH_VAR) {
        Err(VarError::NotPresent) => {
            println!("{} is not defined", CUDA_PATH_VAR);

            for fallback in &platform.fallback_paths {
                println!("Trying default fallback path {:?}", fallback);
                if fallback.exists() {
                    return fallback.to_owned();
                }
            }

            error()
        }
        Err(VarError::NotUnicode(_)) => {
            panic!("Environment variable {} contains non-unicode path", CUDA_PATH_VAR)
        }
        Ok(path) => {
            println!("Using {}={:?}", CUDA_PATH_VAR, path);
            let path = PathBuf::from(path);
            if !path.exists() {
                panic!("Path {}={:?} does not exist", CUDA_PATH_VAR, path);
            }
            return path;
        }
    }
}

fn link_cuda() -> Vec<PathBuf> {
    println!("rerun-if-env-changed={}", CUDA_PATH_VAR);
    println!("rerun-if-env-changed={}", CUDNN_PATH_VAR);

    let platform = platform();

    // find all base dirs
    let mut base_dirs = vec![find_base_dir(&platform)];
    match std::env::var(CUDNN_PATH_VAR) {
        Err(VarError::NotPresent) => {}
        Err(VarError::NotUnicode(_)) => {
            panic!("Environment variable {} contains non-unicode path", CUDNN_PATH_VAR)
        }
        Ok(path) => {
            println!("Using {}={:?}", CUDNN_PATH_VAR, path);
            assert!(std::env::var_os(CUDA_PATH_VAR).is_some(), "Cannot use {} without {}", CUDNN_PATH_VAR, CUDA_PATH_VAR);

            let path = PathBuf::from(path);
            if !path.exists() {
                panic!("Path {}={:?} does not exist", CUDNN_PATH_VAR, path);
            }
            base_dirs.push(path);
        }
    }

    // link dirs
    for path in &base_dirs {
        for suffix in &platform.suffixes_link {
            println!(
                "cargo:rustc-link-search=native={}",
                path.join(suffix).to_str().unwrap()
            );
        }
    }

    // include dirs
    let mut include_paths = vec![];
    for path in &base_dirs {
        for suffix in &platform.suffixes_include {
            include_paths.push(path.join(suffix));
        }
    }
    include_paths
}

fn link_cuda_docs_rs() -> Vec<PathBuf> {
    println!("Running in docs.rs mode, using vendored headers");
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());

    vec![
        manifest_dir.join("doc_headers/cuda_include"),
        manifest_dir.join("doc_headers/cudnn_include"),
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

    // check if this is a docs.rs build
    println!("rerun-if-env-changed=DOCS_RS");
    let is_docs_rs = std::env::var_os("DOCS_RS").is_some();

    // find include directories and set up link search paths
    let include_paths = if is_docs_rs { link_cuda_docs_rs() } else { link_cuda() };

    // add include dirs to builder
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
        .must_use_type("CUresult")
        .must_use_type("cudaError_enum")
        .layout_tests(false)
        // output
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
