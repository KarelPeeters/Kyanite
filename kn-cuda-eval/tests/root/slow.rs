use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::CudaDevice;
use kn_graph::dtype::DType;
use kn_graph::graph::Graph;
use kn_graph::shape;

use crate::root::runner::test_all_exact_graph;
use crate::root::tensor_utils::rng_tensor_f32;

#[test]
#[ignore]
fn layernorm_huge() {
    let mut graph = Graph::new();

    let shape = shape![1, 32, 1048576];
    let input = graph.input(shape, DType::F32);
    let output = graph.layernorm(input, 2, 1e-6);
    graph.output(output);

    // allocate additional buffer to push indices into 64-bit address space
    let buffer = CudaDevice::new(0).alloc(4 * 1024 * 1024 * 1024);

    let mut rng = StdRng::seed_from_u64(0);
    let input_tensor = rng_tensor_f32((1, 32, 1048576), &mut rng);

    test_all_exact_graph(&graph, 0, &[input_tensor], None);

    drop(buffer);
}
