#include "util.h"

__global__ void quantizeKernel(
        int itemsPerThread,
        int length, const float *input, u8 *output
) {
    for (int i = 0; i < itemsPerThread; i++) {
        int index = globalIdx().x * itemsPerThread + i;
        if (index >= length) {
            return;
        }

        float scaled = input[index] * 255.0 + 0.5;
        float clamped = clamp(scaled, 0.0, 255.0);
        output[index] = (u8) clamped;
    }
}

__global__ void unquantizeKernel(
        int itemsPerThread,
        int length, const u8 *input, float *output
) {
    for (int i = 0; i < itemsPerThread; i++) {
        int index = globalIdx().x * itemsPerThread + i;
        if (index >= length) {
            return;
        }

        output[index] = ((float) input[index]) / 255.0;
    }
}

extern "C" {

cudaError quantize(
        cudaStream_t stream,
        int length, const float *input, u8 *output
) {
    int itemsPerThread = 256;
    int threadsPerBlock = 256;

    int itemsPerBlock = itemsPerThread * threadsPerBlock;
    int blockCount = ceil_div(length, itemsPerBlock);

    quantizeKernel<<<blockCount, threadsPerBlock, 0, stream>>>(itemsPerThread, length, input, output);

    return cudaGetLastError();
}

cudaError unquantize(
        cudaStream_t stream,
        int length, const u8 *input, float *output
) {
    int itemsPerThread = 256;
    int threadsPerBlock = 256;

    int itemsPerBlock = itemsPerThread * threadsPerBlock;
    int blockCount = ceil_div(length, itemsPerBlock);

    unquantizeKernel<<<blockCount, threadsPerBlock, 0, stream>>>(itemsPerThread, length, input, output);

    return cudaGetLastError();
}

}