#include "util.cu"

// de-dollar-ify template parameters
const int RANK = $RANK$;
const int OPERANDS = $OPERANDS$;
const int STRIDES_DENSE[RANK] = $STRIDES_DENSE$;
const int STRIDES[OPERANDS][RANK] = $STRIDES$;

__device__ void operation(void *pointers[OPERANDS], int offsets[OPERANDS]) {
    $OPERATION$
}

__global__ void scalar_kernel(
        int batch_size,
        Array<void *, OPERANDS> pointers
) {
    KernelInfo info = kernel_info();

    int size = batch_size * STRIDES_DENSE[0];
    const int itemsPerThread = ceil_div(size, info.thread_count);

    // the main loop, following https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int flat = info.global_thread_id; flat < size; flat += info.thread_count) {
        Array<int, OPERANDS> offsets = flat_index_to_offsets<RANK, OPERANDS>(flat, STRIDES_DENSE, STRIDES);

        operation(pointers.data, &offsets[0]);
    }
}
