use cuda_sys::bindings::{cudaError, cudaStream_t};

// don't warn about cudaError return type
//TODO find a proper solution here
#[allow(improper_ctypes)]
#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn vectorAdd_main();

    pub fn gatherFloat(
        stream: cudaStream_t,
        size: i32,
        indices: *const i32,
        input: *const f32,
        output: *mut f32,
    ) -> cudaError;

    pub fn gather2dAxis1FloatFloat(
        stream: cudaStream_t,
        input_size0: i32,
        input_size1: i32,
        input_stride0: i32,
        input_stride1: i32,
        index_size: i32,
        input: *const f32,
        indices: *const f32,
        output: *mut f32,
    ) -> cudaError;

    pub fn conv8x3Float(
        stream: cudaStream_t,
        batch_size: i32,
        c: i32,
        k: i32,
        input: *const f32,
        filter: *const f32,
        output: *mut f32,
    ) -> cudaError;

    pub fn quantize(
        stream: cudaStream_t,
        batch_size: i32,
        length: i32,
        input: *const f32,
        outputs: *mut *mut u8,
    ) -> cudaError;

    pub fn unquantize(
        stream: cudaStream_t,
        batch_size: i32,
        length: i32,
        inputs: *const *const u8,
        output: *mut f32,
    ) -> cudaError;
}
