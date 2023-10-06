#include "util.cu"

// de-dollar-ify template parameters
typedef $DTYPE$ DType;
typedef $ITYPE$ IType;

const int RANK = $RANK$;

// *CAREFUL* these arrays are actually of length RANK-1, but zero-sized arrays are not allowed in C++ so we pad them
const int KEPT_STRIDES_DENSE[RANK] = $KEPT_STRIDES_DENSE$;
const int KEPT_STRIDES[2][RANK] = $KEPT_STRIDES$;

const int INPUT_AXIS_SIZE = $INPUT_AXIS_SIZE$;
const int INPUT_AXIS_STRIDE = $INPUT_AXIS_STRIDE$;
const int OUTPUT_AXIS_STRIDE = $OUTPUT_AXIS_STRIDE$;
const int INDICES_SIZE_DIV = $INDICES_SIZE_DIV$;
const int INDICES_STRIDE = $INDICES_STRIDE$;
const int OUTPUT_SIZE = $OUTPUT_SIZE$;

// TODO allow scalar fusion on indices and data?
//   think of a proper, general way to handle scalar fusion in all kernels (even with multiple operands?)
__global__ void gather_kernel(
        DType *input,
        IType *indices,
        DType *output
) {
    KernelInfo info = kernel_info();

    int start_output_index = info.global_thread_id;

    for (int output_index = start_output_index; output_index < OUTPUT_SIZE; output_index += info.thread_count) {
        // we could also reorder these (make indices change slowly) to get more linear memory access,
        //   but that turned out to be slower
        int indices_index = output_index % INDICES_SIZE_DIV;
        int kept_index = output_index / INDICES_SIZE_DIV;

        int indices_offset = indices_index * INDICES_STRIDE;
        int index = indices[indices_offset];

        Array<int, 2> io_offsets = flat_index_to_offsets<RANK, 2>(kept_index, KEPT_STRIDES_DENSE, KEPT_STRIDES);
        int input_kept_offset = io_offsets[0];
        int output_kept_offset = io_offsets[1];

        int input_offset = input_kept_offset + index * INPUT_AXIS_STRIDE;

        DType value = 0;
        if (0 <= index && index < INPUT_AXIS_SIZE) {
            value = input[input_offset];
        }

        int output_offset = output_kept_offset + indices_index * OUTPUT_AXIS_STRIDE;
        output[output_offset] = value;
    }
}