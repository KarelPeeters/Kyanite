# Kyanite

[//]: # (TODO write separate main page doc per each crate)
[//]: # (TODO   it doesn't need to be a readme, just a lib.rs doc is fine)

[![Crates.io kn-graph](https://img.shields.io/crates/v/kn-graph?label=kn-graph)](https://crates.io/crates/kn-graph)
[![Crates.io kn-cuda-sys](https://img.shields.io/crates/v/kn-cuda-sys?label=kn-cuda-sys)](https://crates.io/crates/kn-cuda-sys)
[![Crates.io kn-cuda-eval](https://img.shields.io/crates/v/kn-cuda-eval?label=kn-cuda-eval)](https://crates.io/crates/kn-cuda-eval)
[![docs.rs](https://img.shields.io/docsrs/kn-graph)](https://docs.rs/releases/search?query=kyanite+kn-)

A neural network inference library, written in/for Rust. It can run ONNX files either on the CPU or on Nvidia GPUs using cuda/cudnn/cublas/nvrtc.

It is general enough to run all kinds of networks, it has been tested with:
* Simple fully connected networks
* Reset-based CNNs
* Large language models like [LLaMA](https://arxiv.org/abs/2302.13971)
* Image generation models like [Stable Diffusion](https://arxiv.org/abs/2112.10752). For a demo see the `stable_diffusion` example in the `kn-cuda-eval` crate.

The framework is split into three crates:
* `kn-graph`: The core crate, containing the intermediate representation and the CPU executor.
* `kn-cuda-sys`: The Cuda FFI bindings, generated with rust-bindgen.
* `kn-cuda-eval`: The Cuda executor and planner.

## Quick demo

```rust
// Load on onnx file into a graph
let graph = load_graph_from_onnx_path("test.onnx", false)?;
// Optimize the graph
let graph = optimize_graph(&graph, Default::default());
// Render the graph as an svg file
graph_to_svg("test.svg", &graph, false, false)?;

// Build the inputs
let batch_size = 8;
let inputs = [DTensor::F32(Tensor::zeros(IxDyn(&[batch_size, 16])))];

// CPU:
// just evaluate the graph
let outputs: Vec<DTensor> = cpu_eval_graph(&graph, batch_size, &inputs);

// GPU:
// build an executor
let device = Device::new(0);
let mut executor = CudaExecutor::new(device, &graph, batch_size);
// run the executor on the inputs
let outputs: &[DTensor] = executor.evaluate(&inputs);
```

## Details

The typical pipeline is shown in the first figure below. The second figure shows the results of running this pipeline on a simple NN architecture.

![NN inference diagram](./docs/arch_inference.svg)

![conv_bn_sm_flow.svg](./docs/conv_bn_sm_flow.svg)

### Graph IR

[//]: # (TODO separate onnx loading chapter, with some examples and tricks)
[//]: # (TODO link/explain supported ONNX subset)
[//]: # (TODO explain strict orthogonality of IR and why it's better than ONNX)

Central is the _Graph IR_, the intermediate representation for neural network graphs.

The structure is an [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form)-style directed acyclic graph, where nodes are values with a shape, data type and the operation that computes it. These values are abstract, they don't have strides or memory locations yet. 

The operations are similar to those of other frameworks, but are kept as orthogonal as possible. Some example operations: convolution, matmul, reshape, broadcast, slice, unary, binary, reduce, softmax, ... See [the docs](https://docs.rs/kn-graph/latest/kn_graph/graph/enum.Operation.html) for the full list of graph operations.

The graph can be constructed directly in code using the [graph builder api](https://docs.rs/kn-graph/0.2.1/kn_graph/graph/struct.Graph.html), but for convenience an ONNX loader exists. It can read ONNX files and convert the supported subset of operations into those supported by the IR.

Because the graph IR is much more orthogonal than the ONNX specification, many ONNX operations are decomposed into separate steps, some examples:

* ONNX binary operations implicitly broadcast their operands, but this step is a separate operation in the IR.
* ONNX convolution and matmul have a built-in optional bias operand, this also becomes separate broadcast plus binary addition operation.

To figure out if a ONNX operation is supported, check the branches of the top-level match statement in the `visit_node` function in [`load.rs`](https://github.com/KarelPeeters/Kyanite/blob/master/kn-graph/src/onnx/load.rs). Many common operations are already implemented, and adding more operations shouldn't be too hard.

For a larger example of a typical graph, see [stable_diffusion_piece.svg](./docs/stable_diffusion_piece.svg), a small section takes from the start start of the stable diffusion model.

### Optimizer

The graph can optionally be optimized by the _optimizer_. Since the graph is append-only, a new graph is returned. 

The optimizations that are currently implemented are:

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
  
  * This implicitly handles most reshape, broadcast, stride, ... operations.
  * Buffers are also reused if possible, minimizing total memory usage. There is much room for improvement here, currently this is just a single pass algorithm. 

[//]: # (TODO show memory usage graphs?)

* Decide which cudnn/cublas operations to run for convolutions and matmuls If possible, operations are fused together. Some examples:
    * cudnn supports a single "convolution + residual + bias + relu" operation
    * cublas matmuls can include a transpose of either input matrix, and equivalently the output by swapping the inputs.
    * cudnn and cublas operations sometimes include a "scalar" argument that is multiplied by some of the operands

[//]: # (TODO more fusing examples: cublas + scale + transpose, are there others?)

* Compile custom kernels for the remaining scalar and compound operations using an _autokernel_ framework based on [NVRTC (Runtime Compilation)](https://docs.nvidia.com/cuda/nvrtc/index.html).
  
  * The operations handled by *autokernel* are: scalar operations, reduce, softmax, layernorm, gather.
  * Handwritten kernel templates are used, with details such as tensor shapes, strides, scalar operations, ... substituted in before compilation at runtime.
  * More operator fusion happens here
    * Multiple scalar operations get compiled to a single kernel
    * Constant scalars are inlined
    * Some compound kernels support fusing input or output scalar operations


This final operator fusion can be significant and save a lot of redundant transfers to and from main memory. The same performance could be achieved by manually writing kernels for each used combination of operations, but the combinatorial explosion and associated maintenance would be huge.

An example generated scalar kernel with some handwritten clarifying comments is shown below:

<details>
<summary>Example scalar autokernel for residual + batchnorm + relu6</summary>

```cpp
#include "util.cu"

// constants that got inserted into the template
// this scalar operation happens on a tensor of rank 4, with 7 operands
const int RANK = 4;
const int OPERANDS = 7;
const int STRIDES_DENSE[RANK] = {648, 81, 9, 1};
const int STRIDES[OPERANDS][RANK] = {
    // these are full input tensors with normal, dense strides
    {648, 81, 9, 1},
    {648, 81, 9, 1},
    // these values have zero strides for all axes except the channel one,
    //    so these are probably biases and scaling factors
    //    that are broadcast across the other axes
    {0, 1, 0, 0},
    {0, 1, 0, 0},
    {0, 1, 0, 0},
    {0, 1, 0, 0},
    // the output tensor is just another operand
    {648, 81, 9, 1}
};

// the template function, the body of which is generated at runtime
__device__ void operation(void *pointers[OPERANDS], int offsets[OPERANDS]) {
    // all input operand memory locations are cast to the right type
    float *x0 = &((float *) pointers[0])[offsets[0]];
    float *x1 = &((float *) pointers[1])[offsets[1]];
    float *x2 = &((float *) pointers[2])[offsets[2]];
    float *x3 = &((float *) pointers[3])[offsets[3]];
    float *x4 = &((float *) pointers[4])[offsets[4]];
    float *x5 = &((float *) pointers[5])[offsets[5]];
    float *x6 = &((float *) pointers[6])[offsets[6]];
    
    // input operands are loaded
    float y0 = *x0;
    float y1 = *x1;
    
    // this is probably a residual connection
    float y2 = y0 + y1;
    
    // these 4 steps look like they're implementing a batchnorm layer  
    float y3 = *x2;
    float y4 = y2 - y3;
    float y5 = *x3;
    float y6 = y4 / y5;
    float y7 = *x4;
    float y8 = y6 * y7;
    float y9 = *x5;
    float y10 = y8 + y9;
    
    // this implements a relu6 activation function
    float y11 = 6;
    float y12 = min(y10, y11);
    float y13 = (0.0);
    float y14 = max(y12, y13);
    
    // finally the output is stored
    *x6 = y14;
}

// the kernel main function is the same for all scalar kernels
__global__ void scalar_kernel(
        int batch_size,
        Array<void *, OPERANDS> pointers
) {
    KernelInfo info = kernel_info();
    int size = batch_size * STRIDES_DENSE[0];

    // the main loop, following https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int flat = info.global_thread_id; flat < size; flat += info.thread_count) {
        Array<int, OPERANDS> offsets = flat_index_to_offsets<RANK, OPERANDS>(flat, STRIDES_DENSE, STRIDES);
        operation(pointers.data, &offsets[0]);
    }
}
```
</details>