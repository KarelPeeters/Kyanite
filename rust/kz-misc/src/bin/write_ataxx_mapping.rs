use std::fs::File;
use std::io::Write;

use board_game::board::{BoardMoves, BoardSymmetry};
use board_game::games::ataxx::AtaxxBoard;
use board_game::symmetry::{D4Symmetry, Symmetry};
use board_game::util::coord::Coord8;
use internal_iterator::InternalIterator;
use itertools::{enumerate, Itertools};
use serde;

use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::PolicyMapper;

fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("ignored/mapping")?;

    let mut file_valid = File::create("ignored/mapping/ataxx_valid.txt")?;
    let mut file_encode = File::create("ignored/mapping/ataxx_index_to_move_input.txt")?;
    let mut file_syms = File::create("ignored/mapping/ataxx_symmetry.json")?;

    let mut all_syms = vec![];

    for size in 2..=8 {
        let mapper = AtaxxStdMapper::new(size);

        let moves: Vec<_> = AtaxxBoard::all_possible_moves()
            .filter(|mv| mv.valid_for_size(size))
            .collect();

        for (i, &mv) in enumerate(&moves) {
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

        let syms = D4Symmetry::all()
            .iter()
            .map(|&sym| {
                let mut map_mv = vec![];

                for &mv in &moves {
                    let old_index = mapper.move_to_index(mv);
                    let new_mv = AtaxxBoard::empty(size).map_move(sym, mv);
                    let new_index = mapper.move_to_index(new_mv);

                    if old_index >= map_mv.len() {
                        map_mv.resize(old_index + 1, -1);
                    }

                    assert_eq!(map_mv[old_index], -1);
                    map_mv[old_index] = new_index as isize;
                }

                all_numbers += map_mv.len();

                AtaxxSymmetry {
                    transpose: sym.transpose,
                    flip_x: sym.flip_x,
                    flip_y: sym.flip_y,
                    map_mv,
                }
            })
            .collect_vec();
        all_syms.push(syms);
    }

    serde_json::to_writer(&mut file_syms, &all_syms)?;

    Ok(())
}

#[derive(serde::Serialize)]
struct AtaxxSymmetry {
    pub transpose: bool,
    pub flip_x: bool,
    pub flip_y: bool,

    pub map_mv: Vec<isize>,
}
