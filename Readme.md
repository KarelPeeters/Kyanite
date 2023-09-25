# Kyanite

Crates.io published versions:

[![Crates.io kn-graph](https://img.shields.io/crates/v/kn-graph?label=kn-graph)](https://crates.io/crates/kn-graph)
[![Crates.io kn-cuda-sys](https://img.shields.io/crates/v/kn-cuda-sys?label=kn-cuda-sys)](https://crates.io/crates/kn-cuda-sys)
[![Crates.io kn-cuda-eval](https://img.shields.io/crates/v/kn-cuda-eval?label=kn-cuda-eval)](https://crates.io/crates/kn-cuda-eval)

Docs.rs: https://docs.rs/releases/search?query=kyanite+kn-

## Overview

A neural network inference library, written in/for Rust. It can run ONNX files either on the CPU or on GPUs using cuda/cudnn/cublas/nvrtc.

It is general enough to run all kinds of networks, it has been tested with:
* Standard convolutional neural networks, resnets
* Image generation models like [Stable Diffusion](https://arxiv.org/abs/2112.10752). For a demo see the `stable_diffusion` example in the `kn-cuda-eval` crate.
* Large language models like [LLaMA](https://arxiv.org/abs/2302.13971)

The framework is split into three crates:
* `kn-graph`: The core crate, containing the intermediate representation and the CPU executor.
* `kn-cuda-sys`: The Cuda bindings, generated with rust-bindgen.
* `kn-cuda-eval`: The Cuda executor and planner.

## Details

The typical pipeline is shown in the first figure below. The second figure shows the results of running this pipeline on a simple NN architecture.

![NN inference diagram](./docs/arch_inference.svg)

![conv_bn_sm_flow.svg](./docs/conv_bn_sm_flow.svg)

### Graph IR

Central is the _Graph IR_, the intermediate representation for neural network graphs.

The structure is an [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form)-style directed acyclic graph, where nodes are values with a shape, type and the operation that computes it. These values are abstract, they don't have strides or memory locations yet. The operations are similar to those of other frameworks, but are kept as orthogonal as possible. Some example operations: convolution, matmul, reshape, broadcast, slice, unary, binary, reduce, softmax, ... 

The graph can be constructed directly in code, but for convenience an ONNX loader exists. It can read ONNX files and convert the supported subset of operations into those supported by the IR. Many ONNX operations are decomposed into separate steps, some examples:

* ONNX binary operations implicitly broadcast their operands, but this step is a separate operation in the IR.
* ONNX convolution and matmul have a built-in optional bias operand, this also becomes separate broadcast plus binary addition operation.

### Optimizer

The graph can optionally be optimized by the _optimizer_. The optimizations that are currently implemented are:

* Constant folding
* Fusing consecutive affine (bias, scale, batchnorm) operations into a single bias+scale operation.
* Fusing consecutive clamping operations (relu, min, max) into a single min+max operation.
* Strength reduction: replacing division by a constant with multiplication by the inverse constant.
* Recognizing the layernorm template (reduce, subtract, power, reduce, divide) and replacing it with the layernorm operator.

### CPU executor

Finally the graph needs to be executed. There is a simple _CPU executor_ that just directly runs each operation. No major optimizations are attempted here, except for using BLAS routines for matmuls and im2col for convolutions. It's important that this executor is as simple as possible because it serves as the baseline for unit tests that check the correctness of the GPU executor.

### Cuda Executor

The second (and more useful) way to run these graphs is with the _Cuda executor_. This involves running the graph though the _Cuda Planner_, which outputs a predetermined schedule of Cuda operations, and allocates the necessary memory buffers. This is split out as a separate step so this expensive planning step only needs to be carried out once per network architecture, the resulting plan can then be reused many times in the executor.

The planner has the following major responsibilities:

* Determine the memory layout of tensors: the strides and the memory offsets
  
  * This implicitly handles reshape, broadcast, stride, ... operations.
  * We also reuse buffers if possible, minimizing total memory usage.

* Decide which cudnn/cublas operations to run for convolutions and matmuls
  
  * If possible, fuse operations together, eg. cudnn supports a "convolution + residual + bias + relu" operation.

* Compile custom kernels for the remaining scalar and compound operations using an _autokernel_ framework based on [NVRTC (Runtime Compilation)](https://docs.nvidia.com/cuda/nvrtc/index.html).
  
  * The operations handled by *autokernel* are: scalar operations, reduce, softmax, layernorm, gather.
  
  * Handwritten kernel templates are used, with details such as tensor shapes, strides, scalar operations, ... substituted in before compilation at runtime.
  
  * More operator fusion happens here
    
    * Multiple scalar operations get compiled to a single kernel
    
    * Constant scalars are inlined
    
    * Some compound kernels support fusing input or output scalar operations
