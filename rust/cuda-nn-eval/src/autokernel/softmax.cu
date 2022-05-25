#include "util.cu"

// de-dollar-ify template parameters
const int RANK = $RANK$;
const int STATIC_SIZE = $STATIC_SIZE$;
const int SOFTMAX_SIZE = $SOFTMAX_SIZE$;

// *CAREFUL* these arrays are actually of length RANK-1, but zero-sized arrays are not allowed in C++ so we pad them
const int STATIC_DENSE_STRIDES[RANK] = $STATIC_DENSE_STRIDES$;
const int STATIC_STRIDES[2][RANK] = $STATIC_STRIDES$;

const int SOFTMAX_STRIDES[2] = $SOFTMAX_STRIDES$;

// Every block handles a single softmax group.
__global__ void softmax_kernel(
        float *input,
        float *output
) {
    KernelInfo info = kernel_info();

    int static_index = info.global_warp_id;
    if (static_index >= STATIC_SIZE) {
        return;
    }

    Array<int, 2> static_offsets = flat_index_to_offsets<RANK, 2>(static_index, STATIC_DENSE_STRIDES, STATIC_STRIDES);

    float cache[ceil_div(SOFTMAX_SIZE, 32)];
    float max_logit = -1.0 / 0.0;

    // fill cache and calculate max
    for (int i = info.lane_id; i < SOFTMAX_SIZE; i += 32) {
        int offset = static_offsets[0] + i * SOFTMAX_STRIDES[0];
        float curr_logit = input[offset];

        cache[i / 32] = curr_logit;
        max_logit = max(max_logit, curr_logit);
    }

    max_logit = warp_reduce(max_logit, [](float a, float b) { return max(a, b); });
    max_logit = __shfl_sync(FULL_WARP_MASK, max_logit, 0);

    // run exp and calculate sum
    float sum = 0.0;
    for (int i = info.lane_id; i < SOFTMAX_SIZE; i += 32) {
        float tmp = exp(cache[i / 32] - max_logit);
        cache[i / 32] = tmp;
        sum += tmp;
    }

    sum = warp_reduce(sum, [](float a, float b) { return a + b; });
    sum = __shfl_sync(FULL_WARP_MASK, sum, 0);

    // normalize and write to output
    for (int i = info.lane_id; i < SOFTMAX_SIZE; i += 32) {
        int offset = static_offsets[1] + i * SOFTMAX_STRIDES[1];
        float y = cache[i / 32] / sum;
        output[offset] = y;
    }
}