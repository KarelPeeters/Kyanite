#include <cuda_runtime.h>
#include "array.h"

template<typename T, int R>
__global__ void stridedCopyKernel(
        int size,
        Array<int, R> input_strides,
        Array<int, R> output_strides,
        Array<int, R> dense_strides,
        const T *input, T *output
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > size) { return; }

    int input_offset = 0;
    int output_offset = 0;

    for (int r = 0; r < R; r++) {
        int r_i = i / dense_strides[r];
        i = i % dense_strides[r];

        input_offset += r_i * input_strides[r];
        output_offset += r_i * output_strides[r];
    }

    output[output_offset] = input[input_offset];
}

#define BRANCH(r, t) case r: {                                   \
    stridedCopyKernel<t, r><<<blocks, blockSize, 0, stream>>>(   \
        size,                                                    \
        Array<int, r>(input_strides), Array<int, r>(output_strides), Array<int, r>(dense_strides), \
        input, output                                            \
    );                                                           \
    break;                                                       \
}

extern "C" {
cudaError stridedCopyFloat(
        cudaStream_t stream,
        int rank, int size,
        int *input_strides, int *output_strides, int *dense_strides,
        const float *input, float *output
) {
    const int blockSize = 32;
    int blocks = (size + blockSize - 1) / blockSize;

    switch (rank) {
        BRANCH(0, float)
        BRANCH(1, float)
        BRANCH(2, float)
        BRANCH(3, float)
        BRANCH(4, float)
        BRANCH(5, float)
        BRANCH(6, float)
        default:
            return cudaErrorNotSupported;
    }

    return cudaGetLastError();
}
}
