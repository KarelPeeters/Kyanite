use std::fs::File;
use std::io::Write;
use std::ops::ControlFlow;

use board_game::board::BoardMoves;
use board_game::games::ataxx::{AtaxxBoard, Move};
use board_game::util::coord::Coord8;
use internal_iterator::InternalIterator;

use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::PolicyMapper;

fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("ignored/mapping")?;

    let mut file_valid = File::create("ignored/mapping/ataxx_valid.txt")?;
    let mut file_encode = File::create("ignored/mapping/ataxx_index_to_move_input.txt")?;

    for size in 2..=8 {
        let mapper = AtaxxStdMapper::new(size);

        let moves: Vec<_> = AtaxxBoard::all_possible_moves()
            .filter(|mv| mv.valid_for_size(size))
            .collect();

        for (i, mv) in moves.into_iter().enumerate() {
            if i != 0 {
                write!(&mut file_valid, ", ")?;
            }
            write!(&mut file_valid, "{}", mapper.move_to_index(mv))?;
        }

        for i in 0..mapper.policy_len() {
            let mv = mapper.index_to_move(i);

            let (pass, copy_to, jump_from, jump_to) = AtaxxStdMapper::encode_mv_split(mv);
            let int = |c: Option<Coord8>| c.map_or(-1, |c| c.dense_index(size) as isize);

            writeln!(
                &mut file_encode,
                "{} {} {} {}",
                pass as u8,
                int(copy_to),
                int(jump_from),
                int(jump_to)
            )?;
        }

        writeln!(&mut file_valid)?;
        writeln!(&mut file_encode, "=")?;
    }

    Ok(())
}
