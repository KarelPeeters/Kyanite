#include "util.cu"

// de-dollar-ify template parameters
const int KEPT_RANK = $KEPT_RANK$;
const int REDUCED_RANK = $REDUCED_RANK$;

const int KEPT_SIZE = $KEPT_SIZE$;
const int REDUCTION_SIZE = $REDUCTION_SIZE$;

const int KEPT_STRIDES_DENSE[KEPT_RANK] = $KEPT_STRIDES_DENSE$;
const int REDUCED_STRIDES_DENSE[REDUCED_RANK] = $REDUCED_STRIDES_DENSE$;

const int KEPT_STRIDES[2][KEPT_RANK] = $KEPT_STRIDES$;
const int REDUCED_STRIDES[REDUCED_RANK] = $REDUCED_STRIDES$;

typedef $TYPE$ Type;

__device__ Type identity() {
    return $IDENTITY$;
}

__device__ Type reduce(Type curr, Type x) {
    return $OPERATION$;
}

__device__ Type postprocess(Type curr) {
    return $POST_PROCESS$;
}

__global__ void reduce_kernel(
        void *input,
        void *output
) {
    KernelInfo info = kernel_info();
    assert(info.thread_count % info.lane_count == 0);

    // index into the kept axes
    int flat_kept = info.global_thread_id / info.lane_count;
    if (flat_kept >= KEPT_SIZE) {
        return;
    }
    Array<int, 2> kept_offsets = flat_index_to_offsets<KEPT_RANK, 2>(flat_kept, KEPT_STRIDES_DENSE, KEPT_STRIDES);

    // run a partial reduction for each lane
    Type curr = identity();

    for (int flat_red = info.lane_id; flat_red < REDUCTION_SIZE; flat_red += info.lane_count) {
        // additionally, index into the reduced axes
        int red_offset = flat_index_to_offset<REDUCED_RANK>(flat_red, REDUCED_STRIDES_DENSE, REDUCED_STRIDES);
        int total_offset = kept_offsets[0] + red_offset;

        Type x = ((Type *) input)[total_offset];
        curr = reduce(curr, x);
    }

    // reduce over warp
    curr = warp_reduce(curr, reduce);

    // postprocess and write output
    if (info.lane_id == 0) {
        Type result = postprocess(curr);
        ((Type *) output)[kept_offsets[1]] = result;
    }
}