cmake_minimum_required(VERSION 3.20)
project(rust CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(cuda-nn-eval/cuda)
include_directories(cuda-nn-eval/cuda/kernels)

add_executable(rust
        cuda-nn-eval/cuda/kernels/fixed_conv.cu
        cuda-nn-eval/cuda/kernels/gather.cu
        cuda-nn-eval/cuda/kernels/stridedCopy.cu
        cuda-nn-eval/cuda/array.h)
