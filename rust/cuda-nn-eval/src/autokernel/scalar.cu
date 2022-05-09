#include "util.h"

// de-dollar-ify template parameters
const int RANK = $RANK$;
const int OPERANDS = $OPERANDS$;
const int STRIDES_DENSE[RANK] = $STRIDES_DENSE$;
const int STRIDES[OPERANDS][RANK] = $STRIDES$;

__device__ void operation(void *pointers[OPERANDS], int offsets[OPERANDS]) {
    $OPERATION$
}

__global__ void scalar_kernel(
        int batchSize,
        Array<void *, OPERANDS> pointers
) {
    // common startup constants
    const int blockCount = gridDim.x;
    const int threadsPerBlock = blockDim.x;
    const int threadCount = blockCount * threadsPerBlock;

    const int block = blockIdx.x;
    const int thread = threadIdx.x;
    const int global = block * threadsPerBlock + thread;

    int size = batchSize * STRIDES_DENSE[0];
    const int itemsPerThread = ceil_div(size, threadCount);

    // the main loop, following https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int flat = global; flat < size; flat += threadCount) {
        // convert the flat index into a per-operand offset
        int flat_left = flat;
        int offsets[OPERANDS] = {};

        for (int axis = 0; axis < RANK; axis++) {
            int2 result = fast_div(flat_left, STRIDES_DENSE[axis]);
            int axis_index = result.x;
            flat_left = result.y;

            for (int operand = 0; operand < OPERANDS; operand++) {
                offsets[operand] += axis_index * STRIDES[operand][axis];
            }
        }

        // actually run the operation
        operation(pointers.data, &offsets[0]);
    }
}
