__host__ __device__ int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

__device__ int2 fast_div(int a, int b) {
    int q, r;

    // fast path for powers of two (including b=1)
    if (b & (b - 1) == 0) {
        q = a >> __popc(b - 1);
        r = a - q * b;
    } else {
        q = a / b;
        r = a % b;
    }

    return make_int2(q, r);
}

template<typename T, int R>
struct Array {
    T data[R];

    __host__ __device__ T &operator[](int index) {
        return this->data[index];
    }
};

template<int R>
__global__ void foo_kernel(
        int op,

        int size,
        Array<int, R> strides_dense,

        Array<int, R> strides_0, float *ptr_0,
        Array<int, R> strides_1, float *ptr_1,
        Array<int, R> strides_2, float *ptr_2
) {
    int blockCount = gridDim.x;
    int threadsPerBlock = blockDim.x;
    int threadCount = blockCount * threadsPerBlock;

    int block = blockIdx.x;
    int thread = threadIdx.x;
    int global = block * threadsPerBlock + thread;

    int itemsPerThread = ceil_div(size, threadCount);

    for (int index = global; index < size; index += threadCount) {
        int index_left = index;
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;

#pragma unroll
        for (int d = 0; d < R; d++) {
            int2 result = fast_div(index_left, strides_dense[d]);
            int a_d = result.x;
            index_left = result.y;

            i0 += a_d * strides_0[d];
            i1 += a_d * strides_1[d];
            i2 += a_d * strides_2[d];
        }

        if (op) {
            ptr_2[i2] = ptr_0[i0] + ptr_1[i1];
        } else {
            ptr_2[i2] = ptr_0[i0];
        }
    }
}
