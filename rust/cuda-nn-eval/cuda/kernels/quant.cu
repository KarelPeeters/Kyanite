#include "util.h"

__device__ u8 quantize_scalar(float full) {
    float scaled = full * 127.0 + 127.5;
    float clamped = clamp(scaled, 0.0f, 254.0f);
    return (u8) clamped;
}

__device__ float unquantize_scalar(u8 quant) {
    return (((float) quant) - 127.0) / 127.0;
}


__global__ void quantizeKernel(
        int length,
        const float *input, u8 **outputs,
        int itemsPerThread
) {
    int batch_index = blockIdx.x;
    int offset = threadIdx.x * itemsPerThread;

    for (int i = 0; i < itemsPerThread; i++) {
        int index = offset + i;
        if (index >= length) {
            return;
        }
        outputs[batch_index][index] = quantize_scalar(input[batch_index * length + index]);
    }
}

__global__ void unquantizeKernel(
        int length,
        const u8 **inputs, float *output,
        int itemsPerThread
) {
    int batch_index = blockIdx.x;
    int threads = blockDim.x;

    int offset = threadIdx.x;

    for (int i = 0; i < itemsPerThread; i++) {
        int index = offset + i * threads;
        if (index >= length) {
            return;
        }
        output[batch_index * length + index] = unquantize_scalar(inputs[batch_index][index]);
    }
}

extern "C" {

cudaError quantize(
        cudaStream_t stream,
        int batch_size, int length,
        const float *input, u8 **outputs
) {
    int blockCount = batch_size;
    int threadsPerBlock = clamp(length / 64, length, 256);
    int itemsPerThread = ceil_div(length, threadsPerBlock);

    quantizeKernel<<<blockCount, threadsPerBlock, 0, stream>>>(length, input, outputs, itemsPerThread);

    return cudaGetLastError();
}

cudaError unquantize(
        cudaStream_t stream,
        int batch_size, int length,
        const u8 **inputs, float *output
) {
    int blockCount = batch_size;
    int threadsPerBlock = clamp(length / 64, length, 256);
    int itemsPerThread = ceil_div(length, threadsPerBlock);

    unquantizeKernel<<<blockCount, threadsPerBlock, 0, stream>>>(length, inputs, output, itemsPerThread);

    return cudaGetLastError();
}

}