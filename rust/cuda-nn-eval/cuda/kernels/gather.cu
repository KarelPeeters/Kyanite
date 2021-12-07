#include <assert.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void gatherKernel(
        int size,
        const int *indices,
        const T *input, T *output
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > size) { return; }

    output[i] = input[indices[i]];
}

extern "C" {
cudaError gatherFloat(
        cudaStream_t stream,
        int size, const int *indices,
        const float *input, float *output
) {
    int blockSize = 64;
    int blocks = (size + blockSize - 1) / blockSize;

    gatherKernel<float><<<blocks, blockSize, 0, stream>>>(size, indices, input, output);

    return cudaGetLastError();
}
}

__global__ void gather2dAxis1FloatFloatKernel(
        int batch_size, int input_size,
        int input_stride_batch, int input_stride,
        int index_size,
        const float *input, const float *indices, float *output
) {
    int itemsPerThread = (index_size + blockDim.x - 1) / blockDim.x;
    int batch = blockIdx.x;

    for (int i = 0; i < itemsPerThread; i++) {
        int output_index = threadIdx.x * itemsPerThread + i;
        if (output_index >= index_size) {
            break;
        }

        int index = (int) indices[output_index];
        assert(0 <= index && index < input_size);

        float value = input[batch * input_stride_batch + index * input_stride];
        output[batch * index_size + output_index] = value;
    }
}

#include <stdio.h>

extern "C" {
cudaError gather2dAxis1FloatFloat(
        cudaStream_t stream,
        int batch_size, int input_size,
        int input_stride_batch, int input_stride,
        int index_size,
        const float *input, const float *indices, float *output
) {
    int blockCount = batch_size;
    int threadCount = min(64, index_size);

    printf("Launching %d blocks with %d threads\n", blockCount, threadCount);
    fflush(stdout);

    if (blockCount != 0 && threadCount != 0) {
        gather2dAxis1FloatFloatKernel<<<blockCount, threadCount, 0, stream>>>(
                batch_size, input_size,
                input_stride_batch, input_stride,
                index_size,
                input, indices, output
        );
    }

    return cudaGetLastError();
}
}