
__host__ __device__ int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

extern "C" __global__ void
foo_kernel(
        int shape_0, int shape_1,
        int stride0_0, int stride0_1, float *f0,
        int stride1_0, int stride1_1, float *f1,
        int stride2_0, int stride2_1, float *f2
) {
    int blockCount = gridDim.x;
    int threadsPerBlock = blockDim.x;
    int threadCount = blockCount * threadsPerBlock;

    int block = blockIdx.x;
    int thread = threadIdx.x;
    int global = block * threadsPerBlock + thread;

    int size = shape_0 * shape_1;
    int itemsPerThread = ceil_div(size, threadCount);
    int start_index = global * itemsPerThread;

    for (int index = global; index < size; index += threadCount) {
        int a0 = index / shape_1;
        int a1 = index % shape_1;

        int i0 = a0 * stride0_0 + a1 * stride0_1;
        int i1 = a0 * stride1_0 + a1 * stride1_1;
        int i2 = a0 * stride2_0 + a1 * stride2_1;

        f2[i2] = f0[i0] + f1[i1];
    }
}
