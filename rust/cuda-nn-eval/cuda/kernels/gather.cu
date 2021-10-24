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
    const int blockSize = 64;
    int blocks = (size + blockSize - 1) / blockSize;

    gatherKernel<float><<<blocks, blockSize, 0, stream>>>(size, indices, input, output);

    return cudaGetLastError();
}
}
