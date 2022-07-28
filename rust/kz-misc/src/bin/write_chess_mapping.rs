use std::cmp::Reverse;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;
use std::unreachable;

use board_game::chess::{Piece, Square};
use board_game::games::chess::ChessBoard;

use kz_core::mapping::chess::{generate_all_flat_moves_pov, ChessLegacyConvPolicyMapper};
use kz_core::mapping::PolicyMapper;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct FullMove {
    promotion: Option<Reverse<Piece>>,
    knight: bool,
    from: Square,
    to: Square,
}

impl Display for FullMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.promotion {
            None => write!(f, "{}{}", self.from, self.to),
            Some(p) => write!(f, "{}{}={}", self.from, self.to, p.0),
        }
    }
}

fn main() -> std::io::Result<()> {
    std::fs::create_dir_all("ignored/chess_mapping")?;

    let moves = generate_all_flat_moves_pov();

    let output = &mut File::create("ignored/chess_mapping/list.csv")?;
    writeln!(output, "str, flat_i, conv_i, att_from, att_to, att_i")?;

    let mut flat_to_conv = vec![];
    let mut flat_to_att = vec![];
    let mut flat_to_move_input = vec![];

    let dummy_board = ChessBoard::default();

    for (flat_i, &mv_pov) in moves.iter().enumerate() {
        // moves are already POV, and that's the only reason ChessStdMapper needs the board anyway, so this is fine
        // also queen promotion doesn't matter, so just keep none for that
        let conv_i = ChessLegacyConvPolicyMapper.move_to_index(&dummy_board, mv_pov);

        let att_from = mv_pov.get_source().to_index();
        let att_to = match mv_pov.get_promotion() {
            None => mv_pov.get_dest().to_index(),
            Some(p) => {
                let p_i = match p {
                    Piece::Queen => 0,
                    Piece::Rook => 1,
                    Piece::Bishop => 2,
                    Piece::Knight => 3,
                    _ => unreachable!(),
                };

                64 + mv_pov.get_dest().get_file().to_index() * 3 + p_i
            }
        };

        let att_i = att_from * (64 + 8 * 3) + att_to;

        writeln!(
            output,
            "{}, {}, {}, {}, {}, {},",
            mv_pov, flat_i, conv_i, att_from, att_to, att_i
        )?;

        flat_to_conv.push(conv_i);
        flat_to_att.push(att_i);

        let prom = mv_pov.get_promotion();

        flat_to_move_input.push(vec![
            mv_pov.get_source().to_index(),
            mv_pov.get_dest().to_index(),
            0,
            (prom == Some(Piece::Queen)) as usize,
            (prom == Some(Piece::Rook)) as usize,
            (prom == Some(Piece::Bishop)) as usize,
            (prom == Some(Piece::Knight)) as usize,
            (prom == None) as usize,
        ]);
    }

    let mut writer = File::create("ignored/chess_mapping/flat_to_conv.txt")?;
    for &v in &flat_to_conv {
        writeln!(writer, "{}", v)?;
    }

    let mut writer = File::create("ignored/chess_mapping/flat_to_att.txt")?;
    for &v in &flat_to_att {
        writeln!(writer, "{}", v)?;
    }

    let mut writer = File::create("ignored/chess_mapping/flat_to_move_input.txt")?;
    for row in &flat_to_move_input {
        for &v in row {
            write!(writer, "{},", v)?;
        }
        writeln!(writer)?;
    }

    println!("Found {} different full moves", moves.len());
    println!("Without knight promotions: {}", moves.len() - (8 + 7 + 7));
    println!("LC0 move count: 1858");

    Ok(())
}
