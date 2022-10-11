use rand::rngs::StdRng;
use rand::SeedableRng;

use nn_graph::graph::Graph;
use nn_graph::shape;

use crate::root::runner::test_all_exact_graph;
use crate::root::tensor_utils::rng_tensor;

#[test]
#[ignore]
fn layernorm_huge() {
    let mut graph = Graph::new();

    let input = graph.input(shape![1, 32, 1048576]);
    let output = graph.layernorm(input, 2, 1e-6);
    graph.output(output);

    let mut rng = StdRng::seed_from_u64(0);
    let input_tensor = rng_tensor((1, 32, 1048576), &mut rng);

    test_all_exact_graph(&graph, 0, &[input_tensor], None);
}
