#![allow(unreachable_code)]
#![allow(dead_code)]

use std::marker::PhantomData;

use board_game::board::Board;
use board_game::games::chess::{ChessBoard, Rules};

use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::tensor::DeviceTensor;
use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::{InputMapper, MuZeroMapper, PolicyMapper};
use kz_core::muzero::wrapper::MuZeroSettings;
use kz_core::network::common::softmax_in_place;
use kz_core::network::muzero::MuZeroGraphs;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::FpuMode;
use kz_util::display_option;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::OptimizerSettings;

fn main() {
    unsafe { main_impl() }
}

unsafe fn main_impl() {
    let mapper = ChessStdMapper;
    let device = Device::new(0);

    println!("Loading graphs");
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/muzero/max_norm_attached_flip/models_13500";

    let graphs = MuZeroGraphs {
        mapper,
        representation: load_graph_from_onnx_path(format!("{}_representation.onnx", path)),
        dynamics: load_graph_from_onnx_path(format!("{}_dynamics.onnx", path)),
        prediction: load_graph_from_onnx_path(format!("{}_prediction.onnx", path)),
        ph: PhantomData,
    };

    println!("Optimizing graphs");
    let graphs = graphs.optimize(OptimizerSettings::default());

    println!("Fusing graphs & re-optimizing");
    let fused = graphs.fuse(OptimizerSettings::default());

    println!("Building executors");
    let mut exec = fused.executors(device, 1, 1);

    let mut board = ChessBoard::new_without_history_fen(
        "2k5/1pp5/p2r1q2/4p1np/2Q1P1p1/1N4P1/PPP4P/3R3K w - - 0 29",
        Rules::default(),
    );

    println!("Building tree");
    let settings = MuZeroSettings::new(1, UctWeights::default(), false, FpuMode::Parent);
    let visits = 100;
    let tree = settings.build_tree(&board, &mut exec, |tree| tree.root_visits() >= visits);

    println!("{}", tree.display(1, true, usize::MAX, false));

    return;

    println!("Creating executors");
    let mut representation = CudaExecutor::new(device, &graphs.representation, 1);
    let mut dynamics = CudaExecutor::new(device, &graphs.dynamics, 1);
    let mut prediction = CudaExecutor::new(device, &graphs.prediction, 1);

    let moves = ["d1d6", "f6f3", "h1g1", "g5h3", ""];

    let state_shape = fused.state_shape.eval(1);
    let state_tensor = DeviceTensor::alloc_simple(device, state_shape.dims);

    println!("Initial representation");
    let mut input_encoded = vec![];
    mapper.encode_input_full(&mut input_encoded, &board);

    representation.inputs[0].copy_simple_from_host(&input_encoded);
    representation.run_async();
    representation.handles.cudnn.stream().synchronize();
    state_tensor.copy_from(&representation.outputs[0]);

    for (i, mv) in moves.iter().enumerate() {
        println!("\n============");
        println!("Running iteration {} before move {}", i, mv);

        println!("Prediction");

        prediction.inputs[0].copy_from(&state_tensor);
        prediction.run_async();
        prediction.handles.cudnn.stream().synchronize();

        let scalars_tensor = prediction.outputs[0].clone();
        let policy_tensor = prediction.outputs[1].clone();

        let mut scalars = vec![0.0; 5];
        scalars_tensor.copy_simple_to_host(&mut scalars);
        scalars[0] = scalars[0].tanh();
        softmax_in_place(&mut scalars[1..4]);

        let mut policy = vec![0.0; mapper.policy_len()];
        policy_tensor.copy_simple_to_host(&mut policy);
        softmax_in_place(&mut policy);

        println!("Scalars: {:?}", scalars);

        println!("Policy:");
        let mut non_available_mass = 0.0;
        for i in 0..mapper.policy_len() {
            let mv = mapper.index_to_move(&board, i);
            let available = board.is_done() || mv.map_or(false, |mv| board.is_available_move(mv));

            if true || available {
                println!("  {} {} {}", display_option(mv), available, policy[i]);
            }
            if !available {
                non_available_mass += policy[i];
            }
        }

        println!("  * false {}", non_available_mass);

        if mv.is_empty() {
            break;
        }

        println!("Dynamics (playing mv {})", mv);
        let mv = board.parse_move(mv).unwrap();
        let mv_index = mapper.move_to_index(&board, mv).unwrap();

        let mut move_encoded = vec![];
        mapper.encode_mv(&mut move_encoded, mv_index);

        dynamics.inputs[0].copy_from(&state_tensor);
        dynamics.inputs[1].copy_simple_from_host(&move_encoded);
        dynamics.run_async();
        dynamics.handles.cudnn.stream().synchronize();
        state_tensor.copy_from(&dynamics.outputs[0]);

        board.play(mv);
    }
}
