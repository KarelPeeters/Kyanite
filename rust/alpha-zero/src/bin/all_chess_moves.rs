use std::cmp::Reverse;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;
use std::unreachable;

use board_game::games::chess::ChessBoard;
use chess::{Piece, Square};

use alpha_zero::mapping::chess::{ChessLegacyConvPolicyMapper, ChessStdMapper, generate_all_flat_moves_pov};
use alpha_zero::mapping::PolicyMapper;

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

fn main() {
    let moves = generate_all_flat_moves_pov();

    let output = &mut File::create("ignored/moves/list.csv").unwrap();
    writeln!(output, "str, flat_i, conv_i, att_from, att_to, att_i").unwrap();

    let mut flat_to_conv = vec![];
    let mut flat_to_att = vec![];

    let dummy_board = ChessBoard::default();

    for (flat_i, &mv_pov) in moves.iter().enumerate() {
        // moves are already POV, and that's the only reason ChessStdMapper needs the move anyway, so this is fine
        // also queen promotion doesn't matter, so just keep none for that
        let conv_i = ChessLegacyConvPolicyMapper.move_to_index(&dummy_board, mv_pov).unwrap();

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

        writeln!(output, "{}, {}, {}, {}, {}, {},", mv_pov, flat_i, conv_i, att_from, att_to, att_i).unwrap();

        flat_to_conv.push(conv_i);
        flat_to_att.push(att_i);
    }

    std::fs::create_dir_all("ignored/chess_mapping").unwrap();

    write_lines(
        File::create("ignored/chess_mapping/flat_to_conv.txt").unwrap(),
        &flat_to_conv,
    ).unwrap();

    write_lines(
        File::create("ignored/chess_mapping/flat_to_att.txt").unwrap(),
        &flat_to_att,
    ).unwrap();

    println!("Found {} different full moves", moves.len());
    println!("LC0 move count: 1858");
}

fn write_lines(mut writer: impl Write, values: &[usize]) -> std::io::Result<()> {
    for &v in values {
        writeln!(writer, "{}", v)?;
    }

    Ok(())
}
