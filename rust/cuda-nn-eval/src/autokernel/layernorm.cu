#include "util.cu"

// de-dollar-ify template parameters
const int RANK = $RANK$;
const int STATIC_SIZE = $STATIC_SIZE$;
const int NORM_SIZE = $NORM_SIZE$;

const float EPS = $EPS$;
const float ALPHA_0 = $ALPHA_0$;
const float ALPHA_1 = $ALPHA_1$;
const float BETA = $BETA$;

// *CAREFUL* these arrays are actually of length RANK-1, but zero-sized arrays are not allowed in C++ so we pad them
const int STATIC_DENSE_STRIDES[RANK] = $STATIC_DENSE_STRIDES$;
const int STATIC_STRIDES[2][RANK] = $STATIC_STRIDES$;

const int NORM_STRIDES[2] = $NORM_STRIDES$;

// Every block handles a single layernorm group.
// Uses Welford's algorithm to compute the mean and variance
//   (see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).
__global__ void layernorm_kernel(
        float *input0,
        float *input1,
        float *output
) {
    KernelInfo info = kernel_info();

    int static_index = info.global_warp_id;
    if (static_index >= STATIC_SIZE) {
        return;
    }

    Array<int, 2> static_offsets = flat_index_to_offsets<RANK, 2>(static_index, STATIC_DENSE_STRIDES, STATIC_STRIDES);

    float cache[ceil_div(NORM_SIZE, 32)];

    int count = 0;
    float mean = 0.0;
    float m2 = 0.0;

    // fill cache and calculate max
    for (int i = info.lane_id; i < NORM_SIZE; i += 32) {
        int offset = static_offsets[0] + i * NORM_STRIDES[0];

        float curr_raw = ALPHA_0 * input0[offset];
        if (ALPHA_1 != 0.0) {
            curr_raw += ALPHA_1 * input1[offset];
        }

        cache[i / 32] = curr_raw;

        count += 1;
        float delta = curr_raw - mean;
        mean += delta / count;
        m2 += delta * (curr_raw - mean);
    }

    // combine variance and mean between threads
    for (int offset = 16; offset > 0; offset /= 2) {
        int next_count = __shfl_down_sync(FULL_WARP_MASK, count, offset);
        float next_mean = __shfl_down_sync(FULL_WARP_MASK, mean, offset);
        float next_m2 = __shfl_down_sync(FULL_WARP_MASK, m2, offset);

        int prev_count = count;
        count += next_count;

        float delta = next_mean - mean;
        float factor = (float) next_count / (float) count;

        if (factor != factor) {
            factor = 0.0;
        }

        mean += delta * factor;
        m2 += next_m2 + delta * delta * prev_count * factor;
    }

    float var = m2 / count;
    float denom = sqrt(var + EPS);

    // broadcast to all threads
    mean = __shfl_sync(FULL_WARP_MASK, mean, 0);
    denom = __shfl_sync(FULL_WARP_MASK, denom, 0);

    // normalize and write to output
    for (int i = info.lane_id; i < NORM_SIZE; i += 32) {
        int offset = static_offsets[1] + i * NORM_STRIDES[1];
        float x = cache[i / 32];
        float y = (x - mean) / denom;
        output[offset] = BETA * y;
    }
}