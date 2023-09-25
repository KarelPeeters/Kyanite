template<typename T>
__global__ void conv8x3Kernel(
        int batch_size, int c_count, int k_count,
        const T *input,
        const T *filter,
        T *output
) {
    extern __shared__ T buffer[];
    T *filter_buffer = buffer;
    T *input_buffer = &buffer[c_count * k_count * 3 * 3];

    int batch = blockIdx.x;
    int oy = threadIdx.x / 8;
    int ox = threadIdx.x % 8;

    // copy input and filter into shared buffer
    for (int c = 0; c < c_count; c++) {
        for (int k = 0; k < k_count; k++) {
            int input_index = c * 64 + oy * 8 + ox;
            input_buffer[input_index] = input[batch * (c_count * 64) + input_index];

            if (oy < 3 && ox < 3) {
                int filter_index = k * (c_count * 9) + c * 9 + oy * 3 + ox;
                filter_buffer[filter_index] = filter[filter_index];
            }
        }
    }

    __syncthreads();

    // calculate output values
    for (int k = 0; k < k_count; k++) {
        T result = 0;

        for (int c = 0; c < c_count; c++) {
            for (int fy = 0; fy < 3; fy++) {
                for (int fx = 0; fx < 3; fx++) {
                    int iy = oy + fy - 1;
                    int ix = ox + fx - 1;

                    // todo maybe put part of this condition outside of the for loop
                    if (0 <= ix && ix < 8 && 0 <= iy && iy < 8) {
                        T input_value = input_buffer[c * 64 + iy * 8 + ix];
                        T filter_value = filter_buffer[k * (c_count * 9) + c * 9 + fy * 3 + fx];
                        result += input_value * filter_value;
                    }
                }
            }
        }

        output[batch * (64 * k_count) + k * 64 + oy * 8 + ox] = result;
    }
}


extern "C" {
cudaError conv8x3Float(
        cudaStream_t stream,
        int batch_size, int c, int k,
        const float *input,
        const float *filter,
        float *output
) {
    int input_buffer_size = c * 8 * 8;
    int filter_buffer_size = c * k * 3 * 3;

    int buffer_size_bytes = (input_buffer_size + filter_buffer_size) * sizeof(float);

    conv8x3Kernel<float><<<batch_size, 64, buffer_size_bytes, stream>>>(batch_size, c, k, input, filter, output);

    return cudaGetLastError();
}
}
