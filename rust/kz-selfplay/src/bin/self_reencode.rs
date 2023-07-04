use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use board_game::board::{Board, BoardMoves};
use board_game::games::go::{GoBoard, Komi, Rules};
use board_game::pov::ScalarPov;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;
use serde::Deserialize;

use kz_core::mapping::go::GoStdMapper;
use kz_core::mapping::PolicyMapper;
use kz_core::network::ZeroEvaluation;
use kz_core::zero::values::ZeroValuesPov;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::simulation::{Position, Simulation};

#[derive(Deserialize)]
struct DataFile {
    simulations: Vec<DataSimulation>,
}

#[derive(Deserialize)]
struct DataSimulation {
    komi: f32,
    multi_suicide: f32,
    positions: Vec<DataPosition>,
}

#[derive(Deserialize)]
struct DataPosition {
    is_full_search: bool,
    played_mv: i32,
    zero_visits: u64,
    zero_eval_value: Option<f32>,
    zero_eval_wdl: [Option<f32>; 3],
    zero_eval_moves_left: Option<f32>,
    net_eval_value: Option<f32>,
    net_eval_wdl: [Option<f32>; 3],
    net_eval_moves_left: Option<f32>,
    zero_policy_indices: Vec<usize>,
    zero_policy_values: Vec<f32>,
}

fn map_values(value: Option<f32>, wdl: [Option<f32>; 3], moves_left: Option<f32>) -> ZeroValuesPov {
    ZeroValuesPov {
        value: ScalarPov::new(value.unwrap_or(f32::NAN)),
        wdl: WDL::new(
            wdl[0].unwrap_or(f32::NAN),
            wdl[1].unwrap_or(f32::NAN),
            wdl[2].unwrap_or(f32::NAN),
        ),
        moves_left: moves_left.unwrap_or(f32::NAN),
    }
}

fn append_simulation(
    output: &mut BinaryOutput<GoBoard, GoStdMapper>,
    data_sim: &DataSimulation,
    mapper: &GoStdMapper,
) -> std::io::Result<()> {
    let komi_float = data_sim.komi * 15.0;
    let komi = Komi::try_from(komi_float).unwrap();

    let multi_suicide = data_sim.multi_suicide != 0.0;
    let rules = Rules {
        allow_multi_stone_suicide: multi_suicide,
    };

    let mut board = GoBoard::new(9, komi, rules);
    let mut positions = vec![];

    let mut seen_final = false;

    for data_pos in &data_sim.positions {
        assert!(!seen_final);
        if data_pos.played_mv < 0 {
            seen_final = true;
            continue;
        }

        let played_mv_index = data_pos.played_mv.try_into().unwrap();
        let played_mv = mapper.index_to_move(&board, played_mv_index).unwrap();

        let policy: Vec<f32> = board
            .available_moves()
            .unwrap()
            .map(|mv| {
                let mv_index = mapper.move_to_index(&board, mv);

                let p_index = data_pos
                    .zero_policy_indices
                    .iter()
                    .position(|&cand| cand == mv_index)
                    .unwrap();
                let p_value = data_pos.zero_policy_values[p_index];

                p_value
            })
            .collect();

        assert_eq!(policy.len(), data_pos.zero_policy_values.len());

        let zero_eval = ZeroEvaluation {
            values: map_values(
                data_pos.zero_eval_value,
                data_pos.zero_eval_wdl,
                data_pos.zero_eval_moves_left,
            ),
            policy: Cow::Owned(policy.clone()),
        };
        let net_eval = ZeroEvaluation {
            values: map_values(
                data_pos.net_eval_value,
                data_pos.net_eval_wdl,
                data_pos.net_eval_moves_left,
            ),
            // use wrong policy here, just to have something reasonable
            policy: Cow::Owned(policy),
        };

        let new_pos = Position {
            board: board.clone(),
            is_full_search: data_pos.is_full_search,
            played_mv,
            zero_visits: data_pos.zero_visits,
            zero_evaluation: zero_eval,
            net_evaluation: net_eval,
        };

        positions.push(new_pos);
        board.play(played_mv).unwrap();
    }

    let new_sim = Simulation {
        positions,
        final_board: board,
    };
    output.append(&new_sim)?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    // let input_path = r#"C:\Documents\Programming\STTT\kZero\python\main\output.json"#;
    let input_path = r#"\\192.168.0.10\Documents\Karel A0\output.json"#;
    let output_folder = r#"\\192.168.0.10\Documents\Karel A0\loop\go-9\first_own"#;

    let mapper = GoStdMapper::new(9, true);

    let input = BufReader::new(File::open(input_path)?);

    let output_folder = Path::new(output_folder);
    std::fs::create_dir_all(output_folder)?;

    for (i, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        println!("Start processing {i}");

        let data_file = serde_json::from_str::<DataFile>(&line)?;
        let mut output = BinaryOutput::new(output_folder.join(format!("games_{i}")), "go-9", mapper)?;

        for sim in &data_file.simulations {
            append_simulation(&mut output, sim, &mapper)?;
        }

        output.finish()?;
    }

    Ok(())
}
