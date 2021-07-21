use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use internal_iterator::InternalIterator;
use safe_transmute::transmute_to_bytes;
use sttt::board::{Board, BoardAvailableMoves};
use sttt::games::ataxx::{AtaxxBoard, Coord, Move};

use crate::selfplay::{Output, Simulation};

#[derive(Debug)]
pub struct AtaxxBinaryOutput {
    writer: BufWriter<File>,
    next_game_id: usize,
}

impl AtaxxBinaryOutput {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = PathBuf::from(path.as_ref());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .expect("Failed to create directory");
        }
        let file = File::create(&path)
            .expect("Failed to create file");
        let writer = BufWriter::new(file);

        AtaxxBinaryOutput { writer, next_game_id: 0 }
    }
}

const DATA_SIZE: usize = (1 + 3 + 3) + (2 * (1 + 16) * 7 * 7) + (3 * 7 * 7);

impl Output<AtaxxBoard> for AtaxxBinaryOutput {
    fn append(&mut self, simulation: Simulation<AtaxxBoard>) {
        let mut data: Vec<f32> = Vec::new();

        for pos in simulation.iter() {
            data.clear();
            let board = pos.board;

            // game_id
            data.push(self.next_game_id as f32);

            // wdl
            data.push(pos.final_wdl.win);
            data.push(pos.final_wdl.draw);
            data.push(pos.final_wdl.loss);

            data.push(pos.evaluation.wdl.win);
            data.push(pos.evaluation.wdl.draw);
            data.push(pos.evaluation.wdl.loss);

            // policy mask
            extend_with_policy_order(&mut data, |mv| {
                board.is_available_move(mv) as u8 as f32
            });

            // policy
            let policy = pos.evaluation.policy;
            let moves: Vec<Move> = board.available_moves().collect();

            extend_with_policy_order(&mut data, |mv| {
                if board.is_available_move(mv) {
                    let index = moves.iter().position(|&cand| cand == mv).unwrap();
                    policy[index]
                } else {
                    0.0
                }
            });

            // board state
            let (next_tiles, other_tiles) = board.tiles_pov();
            data.extend(Coord::all().map(|c| next_tiles.has(c) as u8 as f32));
            data.extend(Coord::all().map(|c| other_tiles.has(c) as u8 as f32));
            data.extend(Coord::all().map(|c| board.gaps().has(c) as u8 as f32));

            //actually write to the file
            assert_eq!(DATA_SIZE, data.len());
            self.writer.write_all(transmute_to_bytes(&data))
                .expect("Failed to write to file");
        }
        self.next_game_id += 1;
        self.writer.flush().expect("Failed to flush");
    }
}

pub const FROM_DX_DY: [(i8, i8); 16] = [
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
    (-2, -1), (2, -1),
    (-2, 0), (2, 0),
    (-2, 1), (2, 1),
    (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
];

/// Call f for each possible move, and push the returned values to output.
/// Will push `(1+16)*7*7` values in total, using `0.0` for nonsense moves (jumps near the edge).
fn extend_with_policy_order(output: &mut Vec<f32>, f: impl Fn(Move) -> f32) {
    for to in Coord::all() {
        output.push(f(Move::Copy { to }))
    }

    // Here we have to be careful to push 16 values every time,
    // and to keep the "from" axis first for consistency with the copy moves.
    for &(dx, dy) in &FROM_DX_DY {
        for to in Coord::all() {
            let fx = to.x() as i32 + dx as i32;
            let fy = to.y() as i32 + dy as i32;

            if (0..7).contains(&fx) && (0..7).contains(&fy) {
                let from = Coord::from_xy(fx as u8, fy as u8);
                output.push(f(Move::Jump { from, to }))
            } else {
                output.push(0.0);
            }
        }
    }
}
