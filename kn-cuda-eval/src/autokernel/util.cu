// fix for missing int types header on NVRTC
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

static_assert(sizeof(int8_t) == 1, "int8_t is not 8 bits");
static_assert(sizeof(int16_t) == 2, "int16_t is not 16 bits");
static_assert(sizeof(int32_t) == 4, "int32_t is not 32 bits");
static_assert(sizeof(int64_t) == 8, "int64_t is not 64 bits");

constexpr __device__ int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

struct KernelInfo {
    int block_count;
    int threads_per_block;
    int thread_count;

    int block_id;
    int thread_id;
    int global_thread_id;

    int warp_id;
    int global_warp_id;
    int lane_id;

    int lane_count;
};

__device__ KernelInfo kernel_info() {
    KernelInfo info;

    info.block_count = gridDim.x;
    info.threads_per_block = blockDim.x;
    info.thread_count = blockDim.x * gridDim.x;

    info.block_id = blockIdx.x;
    info.thread_id = threadIdx.x;
    info.global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    info.warp_id = info.thread_id / 32;

    int warps_per_block = ceil_div(info.threads_per_block, 32);
    info.global_warp_id = info.block_id * warps_per_block + info.thread_id / 32;
    info.lane_id = threadIdx.x % 32;

    info.lane_count = 32;

    return info;
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

template<int RANK, int OPERANDS>
__device__ Array<int, OPERANDS> flat_index_to_offsets(
        int flat,
        const int strides_dense[RANK],
        const int strides[OPERANDS][RANK]
) {
    int flat_left = flat;
    Array<int, OPERANDS> offsets = {};

    for (int axis = 0; axis < RANK; axis++) {
        int2 result = fast_div(flat_left, strides_dense[axis]);
        int axis_index = result.x;
        flat_left = result.y;

        for (int operand = 0; operand < OPERANDS; operand++) {
            offsets[operand] += axis_index * strides[operand][axis];
        }
    }

    return offsets;
}

template<int RANK>
__device__ int flat_index_to_offset(
        int flat,
        const int strides_dense[RANK],
        const int strides[RANK]
) {
    Array<int, 1> offsets = flat_index_to_offsets<RANK, 1>(flat, strides_dense, (int (*)[RANK]) strides);
    return offsets[0];
}

const unsigned int FULL_WARP_MASK = 0xffffffff;

// Reduce a value across all threads in a block, towards the first thread.
// To get the value in all threads, use `__shfl(value, 0)`.
// see https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
template<typename T, typename F>
__device__ T warp_reduce(T value, F f) {
    T result = value;

    for (int offset = 16; offset > 0; offset /= 2) {
        T next = __shfl_down_sync(FULL_WARP_MASK, result, offset);
        result = f(result, next);
    }

    return result;
}

template<typename T, typename U>
__device__ T bit_cast(U value) {
    // don't use a union, that would be UB
    static_assert(sizeof(T) == sizeof(U), "bit_cast requires types of equal size");
    T result;
    memcpy(&result, &value, sizeof(T));
    return result;
}