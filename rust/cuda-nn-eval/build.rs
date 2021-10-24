fn main() {
    // based on https://github.com/termoshtt/link_cuda_kernel

    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("kernels/kernels.cu")
        .compile("libkernels.a")

    // we're not linking anything extra here since cuda stuff should already be linked by cuda-sys
}