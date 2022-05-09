__host__ __device__

int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

__device__ int2

fast_div(int a, int b) {
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

    __device__ T
    &

    operator[](int index) {
        return this->data[index];
    }
};
