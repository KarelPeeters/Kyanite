use image::{GenericImageView, ImageBuffer, Luma};
use ndarray::s;
use ndarray_stats::QuantileExt;
use rand::thread_rng;

use alpha_zero::games::ataxx_cpu_network::AtaxxCPUNetwork;
use alpha_zero::games::ataxx_output::visit_policy_order;
use alpha_zero::games::ataxx_utils::encode_input;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use cuda_nn_eval::cpu_executor::CpuExecutor;
use cuda_nn_eval::fuser::FusedValueInfo;
use cuda_nn_eval::graph::Graph;
use cuda_nn_eval::onnx::load_onnx_graph;

const NON_RES_PADDING: usize = 4;

fn plot_network_activations(graph: &Graph, executor: &mut CpuExecutor, board: &AtaxxBoard) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let input = encode_input(&[board.clone()]);
    let mut output_wdl = vec![0.0; 3];
    let mut output_policy = vec![0.0; 17 * 7 * 7];
    executor.evaluate(&[&input], &mut [&mut output_wdl, &mut output_policy]);

    let mut next_y = 0;
    let mut width_used = 0;

    let mut image = ImageBuffer::from_pixel(10_000, 10_000, Luma([100]));

    let fused_graph = &executor.fused_graph();
    for fused_value in fused_graph.schedule() {
        let fused_info = fused_graph[fused_value];
        let value = fused_info.value();

        match fused_info {
            FusedValueInfo::Constant(_) => continue,
            FusedValueInfo::Input(_) | FusedValueInfo::FusedOperation { .. } => {
                let [n, c, w, h] = graph[fused_info.value()].shape;
                let mut data = executor.buffers().get(&fused_value).unwrap().clone();
                assert_eq!(1, n);

                let is_res = matches!(fused_info, FusedValueInfo::FusedOperation { res_input: Some(_), .. });
                if is_res {
                    next_y -= NON_RES_PADDING;
                }

                let is_output = graph.outputs().contains(&value);
                let is_wdl = is_output && c == 3 && w == 1 && h == 1;
                let is_policy = is_output && c == 17 && w == 7 && h == 7;

                // mask policy
                if is_policy {
                    let mut next_index = 0;
                    visit_policy_order(|mv| {
                        if mv.map_or(true, |mv| !board.is_available_move(mv)) {
                            data.as_slice_mut().unwrap()[next_index] = f32::NEG_INFINITY;
                        }
                        next_index += 1;
                    });
                }

                // logit -> prob conversion for outputs
                if is_wdl || is_policy {
                    data.map_inplace(|x| *x = x.exp());
                    let total = data.sum();
                    data /= total;
                }

                for ci in 0..c as usize {
                    let max = if is_wdl || is_policy || (w == 1 && h == 1) {
                        *data.max().unwrap()
                    } else {
                        *data.slice(s![0, ci, .., ..]).max().unwrap()
                    };

                    assert!(max.is_finite() && max >= 0.0);
                    let max = if max != 0.0 { max } else { 1.0 };

                    for wi in 0..w as usize {
                        for hi in 0..h as usize {
                            let value = data[(0, ci, wi, hi)];
                            assert!((0.0..=max).contains(&value), "failed range check: 0.0 <= {} <= {}", value, max);

                            let value_int = (value / max * 255.0) as u8;

                            let pixel = Luma([value_int]);
                            image.put_pixel((ci * (w + 1) as usize + wi) as u32, (next_y + hi) as u32, pixel)
                        }
                    }
                }

                next_y += h as usize + NON_RES_PADDING + 1;
                width_used = std::cmp::max(width_used, c * (w + 1))
            }
        }
    }

    image.view(0, 0, width_used as u32, next_y as u32)
        .to_image()
}

fn main() {
    std::fs::create_dir_all("ignored/activations").unwrap();

    let path = "../data/derp/test_loop/gen_240/model_1_epochs.onnx";
    let graph = load_onnx_graph(path, 1);

    let mut network = AtaxxCPUNetwork::load(path, 1);
    let mut executor = CpuExecutor::new(&graph);

    let mut board = AtaxxBoard::new_without_gaps();

    for i in 0.. {
        if board.is_done() { break; }

        let image = plot_network_activations(&graph, &mut executor, &board);
        image.save(format!("ignored/activations/image_{}.png", i)).unwrap();

        let mut rng = thread_rng();
        let tree = zero_build_tree(&board, 1000, ZeroSettings::new(2.0, true), &mut network, &mut rng, || false);
        println!("{}", tree.display(1));

        board.play(tree.best_move())
    }
}

