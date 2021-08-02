use std::io::{BufWriter, Write};

use crossbeam::channel::Receiver;

use board_game::board::Board;

use crate::new_selfplay::core::GeneratorUpdate;
use crate::selfplay::Output;

pub fn collector_main<B: Board, O: Output<B>>(
    writer: BufWriter<impl Write>,
    games_per_file: usize,
    output_folder: &str,
    output: impl Fn(&str) -> O,
    update_receiver: Receiver<GeneratorUpdate<B>>,
) {
    // TODO figure this out from the currently existing files instead
    //  maybe write to a separate file first, and then rename to the final file, so we never have unfinished files
    //  that are assumed to be finished
    let mut curr_i = 0;
    let mut curr_output = output(&format!("{}/games_{}.bin", output_folder, curr_i));
    let mut curr_game_count = 0;

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,
            GeneratorUpdate::FinishedSimulation(simulation) => {
                println!("Finished simulation with board \n{}", simulation.positions.last().unwrap().board);

                curr_output.append(simulation);
                curr_game_count += 1;

                println!("{}", curr_game_count);

                if curr_game_count >= games_per_file {
                    curr_i += 1;
                    curr_game_count = 0;
                    curr_output = output(&format!("{}/games_{}.bin", output_folder, curr_i))
                }
            }
            update @ GeneratorUpdate::Progress { .. } => {
                println!("Progress update: {:?}", update);

                //TODO aggregate updates and send them over tcp in large batches
            }
        }
    }
}
