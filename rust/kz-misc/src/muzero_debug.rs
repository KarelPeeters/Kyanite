use std::iter;

use board_game::board::Board;

use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::quant::{BatchQuantizer, QuantizedStorage};
use cuda_sys::wrapper::handle::{CudaStream, Device};
use kz_core::mapping::BoardMapper;
use kz_core::muzero::wrapper::MuZeroSettings;
use kz_core::network::common::{softmax, softmax_in_place, zero_values_from_scalars};
use kz_core::network::muzero::{ExpandArgs, MuZeroGraphs, RootArgs};
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::FpuMode;
use kz_util::display::display_option;
use nn_graph::graph::SliceRange;
use nn_graph::optimizer::OptimizerSettings;

pub fn muzero_debug_utility<B: Board, M: BoardMapper<B>>(
    path: &str,
    start: B,
    mapper: M,
    moves: &[B::Move],
    unroll: bool,
    unroll_fused: bool,
    tree_visits: Option<u64>,
) {
    unsafe {
        muzero_debug_utility_inner(path, start, mapper, moves, unroll, unroll_fused, tree_visits);
    }
}

unsafe fn muzero_debug_utility_inner<B: Board, M: BoardMapper<B>>(
    path: &str,
    start: B,
    mapper: M,
    moves: &[B::Move],
    unroll: bool,
    unroll_fused: bool,
    tree_visits: Option<u64>,
) {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    println!("Loading graphs");
    let graphs = MuZeroGraphs::load(&path, mapper);
    let state_shape = graphs.info.state_shape(mapper).eval(1);
    let saved_state_shape = graphs.info.state_saved_shape(mapper).eval(1);
    let state_slice = SliceRange::simple(0, graphs.info.state_channels_saved);

    println!("Optimizing graphs");
    let graphs = graphs.optimize(OptimizerSettings::default());

    // auxiliary buffers and helpers
    let full_state_tensor = DeviceTensor::alloc_simple(device, state_shape.dims.clone());
    let saved_state_quant = QuantizedStorage::alloc(device, saved_state_shape.size());
    let saved_state_tensor = DeviceTensor::alloc_simple(device, saved_state_shape.dims.clone());

    let mut full_state_buffer = vec![f32::NAN; state_shape.size()];
    let mut saved_state_buffer = vec![f32::NAN; saved_state_shape.size()];
    let mut quantizer = BatchQuantizer::new(device, 1);

    if unroll {
        chapter("Unrolling");

        let mut board = start.clone();

        println!("Building separate executors");
        let mut representation = CudaExecutor::new(device, &graphs.representation, 1);
        let mut dynamics = CudaExecutor::new(device, &graphs.dynamics, 1);
        let mut prediction = CudaExecutor::new(device, &graphs.prediction, 1);

        let mut input_encoded = vec![];
        mapper.encode_input_full(&mut input_encoded, &board);
        println!("root input: {input_encoded:?}");

        representation.inputs[0].copy_simple_from_host(&input_encoded);
        representation.run_async();
        representation.handles.cudnn.stream().synchronize();
        full_state_tensor.copy_from(&representation.outputs[0]);

        let mut moves = moves.into_iter();

        loop {
            let mv = moves.next();

            // process state
            quantizer.launch_quantize(
                &stream,
                &full_state_tensor.slice(1, state_slice),
                iter::once(&saved_state_quant),
            );
            quantizer.launch_unquantize(&stream, iter::once(&saved_state_quant), &saved_state_tensor);
            stream.synchronize();
            full_state_tensor.copy_simple_to_host(&mut full_state_buffer);
            saved_state_tensor.copy_simple_to_host(&mut saved_state_buffer);

            println!("full state: {:?}", full_state_buffer);
            println!("saved state: {:?}", saved_state_buffer);

            // prediction
            prediction.inputs[0].copy_from(&full_state_tensor);
            prediction.run_async();
            prediction.handles.cudnn.stream().synchronize();

            let scalars_tensor = prediction.outputs[0].clone();
            let policy_tensor = prediction.outputs[1].clone();

            let mut scalars = vec![0.0; 5];
            scalars_tensor.copy_simple_to_host(&mut scalars);
            let values = zero_values_from_scalars(&scalars);

            let mut policy = vec![0.0; mapper.policy_len()];
            policy_tensor.copy_simple_to_host(&mut policy);
            softmax_in_place(&mut policy);

            println!("scalars: {:?}", scalars);
            println!("values: {:?}", values);

            println!("policy: {:?}", policy);
            let mut non_available_mass = 0.0;
            let mut available_mass = 0.0;

            println!("Detailed policy: index mv available p");
            for i in 0..mapper.policy_len() {
                let mv = mapper.index_to_move(&board, i);
                let available = board.is_done() || mv.map_or(false, |mv| board.is_available_move(mv));

                println!("  {} {} {} {}", i, display_option(mv), available, policy[i]);

                if available {
                    available_mass += policy[i];
                } else {
                    non_available_mass += policy[i];
                }
            }

            println!("Total policy:");
            println!("  available {}", available_mass);
            println!("  non-available {}", non_available_mass);

            let &mv = match mv {
                Some(mv) => mv,
                None => break,
            };

            // encode move
            let mv_index = mapper.move_to_index(&board, mv).unwrap();
            println!("Dynamics with move {} {}", mv_index, mv);

            let mut move_encoded = vec![];
            mapper.encode_mv(&mut move_encoded, mv_index);
            dynamics.inputs[1].copy_simple_from_host(&move_encoded);

            println!("encoded move: {:?}", move_encoded);

            // run dynamics
            dynamics.inputs[0].copy_from(&saved_state_tensor);
            dynamics.handles.cudnn.stream().synchronize();
            dynamics.run_async();
            dynamics.handles.cudnn.stream().synchronize();
            full_state_tensor.copy_from(&dynamics.outputs[0]);

            // actually play the move on the given board
            board.play(mv);
        }
    }

    println!("Fusing graphs");
    let fused = graphs.fuse(OptimizerSettings::default());

    println!("Building fused executors");
    let mut root_exec = fused.root_executor(device, 1);
    let mut expand_exec = fused.expand_executor(device, 1);

    if unroll_fused {
        chapter("Fused unrolling");

        let mut curr_quant = QuantizedStorage::alloc(device, saved_state_shape.size());
        let mut next_quant = QuantizedStorage::alloc(device, saved_state_shape.size());

        let root_args = RootArgs {
            board: start.clone(),
            output_state: curr_quant.clone(),
        };
        let mut eval = root_exec.eval_root(&[root_args]).remove(0);

        let mut moves = moves.iter();

        loop {
            let mut curr_board = start.clone();

            quantizer.launch_unquantize(&stream, iter::once(&curr_quant), &saved_state_tensor);
            stream.synchronize();
            saved_state_tensor.copy_simple_to_host(&mut saved_state_buffer);

            println!("values: {:?}", eval.values);
            println!("policy: {:?}", softmax(&eval.policy_logits));

            println!("saved state: {:?}", saved_state_buffer);

            match moves.next().copied() {
                None => break,
                Some(mv) => {
                    let mv_index = mapper.move_to_index(&curr_board, mv).unwrap();
                    curr_board.play(mv);

                    let expand_args = ExpandArgs {
                        state: curr_quant.clone(),
                        move_index: mv_index,
                        output_state: next_quant.clone(),
                    };

                    eval = expand_exec.eval_expand(&[expand_args]).remove(0);
                    std::mem::swap(&mut curr_quant, &mut next_quant);
                }
            }
        }
    }

    if let Some(tree_visits) = tree_visits {
        chapter("Tree");

        println!("Building tree");
        let top_moves = 100;
        let settings = MuZeroSettings::new(1, UctWeights::default(), false, FpuMode::Parent, top_moves);
        let tree = settings.build_tree(&start, u32::MAX, &mut root_exec, &mut expand_exec, |tree| {
            tree.root_visits() >= tree_visits
        });

        println!("Finished tree");
        println!("{}", tree.display(3, true, 10, true));

        println!("Extracting quantized states");
        let mut moves = moves.iter().copied();
        let mut curr_node = 0;
        let mut curr_board = start.clone();

        loop {
            if let Some(inner) = tree[curr_node].inner.as_ref() {
                quantizer.launch_unquantize(&stream, iter::once(&inner.state), &saved_state_tensor);
                stream.synchronize();
                saved_state_tensor.copy_simple_to_host(&mut saved_state_buffer);

                println!("saved state: {:?}", saved_state_buffer);

                if let Some(mv) = moves.next() {
                    let mv_index = mapper.move_to_index(&curr_board, mv);
                    let child_node = inner
                        .children
                        .iter()
                        .find(|&n| tree[n].last_move_index == mv_index)
                        .unwrap();
                    curr_board.play(mv);
                    curr_node = child_node;
                    continue;
                }
            } else {
                if let Some(mv) = moves.next() {
                    let mv_index = mapper.move_to_index(&curr_board, mv);
                    println!("Tree not deep enough, stopped before move {:?} {}", mv_index, mv);
                }
            }

            break;
        }

        println!("Principal variation:");
        let mut test_board = start.clone();
        for mv_index in tree.principal_variation(10) {
            let mv = mapper.index_to_move(&test_board, mv_index.unwrap());
            println!("  {:?} {}", mv_index, display_option(mv));
            if let Some(mv) = mv {
                if !test_board.is_done() && test_board.is_available_move(mv) {
                    test_board.play(mv);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
}

fn chapter(name: &str) {
    println!("\n{}\n====================", name);
}
