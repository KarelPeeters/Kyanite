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

    __device__ T &operator[](int index) {
        return this->data[index];
    }
};

__device__ void operation(float *x[$OPERANDS$]) {
    $OPERATION$;
}

__global__ void scalar_kernel(
        Array<float *, $OPERANDS$> pointers
) {
    // de-dollar-ify parameters
    const int size = $SIZE$;
    const int rank = $RANK$;
    const int operands = $OPERANDS$;
    const int strides_dense[$RANK$] = $STRIDES_DENSE$;
    const int strides[$OPERANDS$][$RANK$] = $STRIDES$;

    // common startup constants
    const int blockCount = gridDim.x;
    const int threadsPerBlock = blockDim.x;
    const int threadCount = blockCount * threadsPerBlock;

    const int block = blockIdx.x;
    const int thread = threadIdx.x;
    const int global = block * threadsPerBlock + thread;

    const int itemsPerThread = ceil_div(size, threadCount);

    // the main loop, following https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int flat = global; flat < size; flat += threadCount) {
        // convert the flat index into a per-operand offset
        int flat_left = flat;
        int offsets[operands] = {};

        for (int axis = 0; axis < rank; axis++) {
            int2 result = fast_div(flat_left, strides_dense[axis]);
            int axis_index = result.x;
            flat_left = result.y;

            for (int operand = 0; operand < operands; operand++) {
                offsets[operand] += axis_index * strides[operand][axis];
            }
        }

        // get a pointer into each operand
        float *x[operands];
        for (int operand = 0; operand < operands; operand++) {
            x[operand] = &pointers[operand][offsets[operand]];
        }

        // actually run the operation
        operation(x);
    }
}
