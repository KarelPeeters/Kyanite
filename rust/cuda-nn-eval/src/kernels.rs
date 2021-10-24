use cuda_sys::bindings::{cudaError, cudaStream_t};

// don't warn about cudaError return type
//TODO find a proper solution here
#[allow(improper_ctypes)]
#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn vectorAdd_main();

    pub fn stridedFloatCopy(
        stream: cudaStream_t,
        rank: i32, size: i32,
        input_strides: *const i32, output_strides: *const i32, dense_strides: *const i32,
        input: *const f32, output: *mut f32,
    ) -> cudaError;
}