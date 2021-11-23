use std::cmp::Reverse;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::unreachable;
use std::io::Write;

use board_game::board::{Board, BoardAvailableMoves};
use board_game::games::chess::ChessBoard;
use chess::{ChessMove, Piece, Square};
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::thread_rng;
use alpha_zero::mapping::chess::{ChessStdMapper, square_pov};
use alpha_zero::mapping::PolicyMapper;

use alpha_zero::util::PrintThroughput;

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
    let separate_queen_promotion = true;
    let separate_castling = false;

    let mut moves: HashMap<FullMove, u64> = Default::default();
    let mut rng = thread_rng();

    let mut pt = PrintThroughput::new("games");
    for _ in 0..10_000 {
        let mut board = ChessBoard::default();

        while !board.is_done() {
            board.available_moves().for_each(|mv| {
                let full_mv_pov = FullMove {
                    promotion: mv.get_promotion().filter(|&p| separate_queen_promotion || p != Piece::Queen).map(|p| Reverse(p)),
                    knight: board.inner().piece_on(mv.get_source()) == Some(Piece::Knight),
                    from: square_pov(board.inner().side_to_move(), mv.get_source()),
                    to: square_pov(board.inner().side_to_move(), mv.get_dest()),
                };

                *moves.entry(full_mv_pov).or_insert(0) += 1;
            });
            board.play(board.random_available_move(&mut rng));
        }

        if pt.update(1) {
            println!("wip move count: {}", moves.len());
        }
    }

    //TODO think about fully separating moves by piece and type
    // (pawn single,pawn double,rook,bishop,queen,promotion,knight,long castle, short castle)
    let mut moves = moves.iter().collect_vec();
    moves.sort_by_key(|(&mv, _)| mv);

    let output = &mut File::create("ignored/chess_moves.csv").unwrap();
    writeln!(output, "str, flat_i, conv_i, att_from, att_to").unwrap();

    let dummy_board = ChessBoard::default();

    for (flat_i, (&mv_pov, _)) in moves.iter().enumerate() {
        // moves are already POV, and that's the only reason ChessStdMapper needs the move anyway, so this is fine
        // also queen promotion doesn't matter, so just keep none for that
        let chess_mv_pov = ChessMove::new(mv_pov.from, mv_pov.to, mv_pov.promotion.map(|p| p.0));
        let conv_i = ChessStdMapper.move_to_index(&dummy_board, chess_mv_pov).unwrap();

        let att_from = mv_pov.from.to_index();
        let att_to = match mv_pov.promotion {
            None => mv_pov.to.to_index(),
            Some(Reverse(p)) => {
                let p_i = match p {
                    Piece::Queen => 0,
                    Piece::Rook => 1,
                    Piece::Bishop => 2,
                    Piece::Knight => 3,
                    _=> unreachable!(),
                };

                64 + mv_pov.to.get_file().to_index() * 3 + p_i
            }
        };

        writeln!(output, "{}, {}, {}, {}, {},", mv_pov, flat_i, conv_i, att_from, att_to).unwrap();
    }

    println!("Found {} different full moves", moves.len());
    println!("LC0 move count: 1858");
}
