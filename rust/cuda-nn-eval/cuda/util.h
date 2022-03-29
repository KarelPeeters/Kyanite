#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

__host__ __device__

int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

template<class T>
__host__ __device__

T clamp(T x, T min_value, T max_value) {
    return min(max(x, min_value), max_value);
}

__device__ uint3

globalIdx() {
    return make_uint3(
            blockIdx.x * gridDim.x + threadIdx.x,
            blockIdx.y * gridDim.y + threadIdx.y,
            blockIdx.z * gridDim.z + threadIdx.z
    );
}