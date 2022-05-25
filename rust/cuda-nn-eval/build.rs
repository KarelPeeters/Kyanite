use std::ffi::OsStr;
use std::path::Path;

fn main() {
    // based on https://github.com/termoshtt/link_cuda_kernel

    let kernels_path = "cuda/kernels";
    println!("cargo:rerun-if-changed={}", kernels_path);

    // this is stupid, there's way to much unwrapping here
    let kernels_path = Path::new(kernels_path);
    let mut files = vec![];
    for f in std::fs::read_dir(kernels_path).unwrap() {
        let f = f.unwrap();
        let name = f.file_name();
        if f.file_type().unwrap().is_file() && Path::new(&name).extension() == Some(OsStr::new("cu")) {
            files.push(kernels_path.join(name))
        }
    }

    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .debug(false)
        .opt_level(3)
        // .flag("-gencode")
        // .flag("arch=compute_61,code=sm_61")
        .include("cuda")
        .files(files)
        .compile("libkernels.a")

    // we're not linking anything extra here since cuda stuff should already be linked by cuda-sys
}
