use std::cmp::max;
use std::fmt::Debug;

use board_game::games::chess::ChessBoard;
use chess::{Board, ChessMove, File, Piece, Rank, Square};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::util::IndexOf;

//TODO try different embeddings discussed in Discord
//TODO AlphaZero also adds history, why?
#[derive(Debug, Copy, Clone)]
pub struct ChessStdMapper;

const INPUT_CHANNELS: usize = 2 + 2 * 6 + 4 + 1;

impl InputMapper<ChessBoard> for ChessStdMapper {
    const INPUT_SHAPE: [usize; 3] = [INPUT_CHANNELS, 8, 8];

    fn append_board_to(&self, result: &mut Vec<f32>, board: &ChessBoard) {
        let inner = board.inner();

        //absolute reference for the current player
        for color in chess::ALL_COLORS {
            result.extend(std::iter::repeat((inner.side_to_move() == color) as u8 as f32).take(8 * 8));
        }

        // everything else is from the next player's POV
        let pov_colors = [inner.side_to_move(), !inner.side_to_move()];

        //pieces
        for &color in &pov_colors {
            for piece in chess::ALL_PIECES {
                for square in chess::ALL_SQUARES {
                    let value = inner.color_on(square) == Some(color) && inner.piece_on(square) == Some(piece);
                    result.push(value as u8 as f32);
                }
            }
        }

        //castling rights
        for &color in &pov_colors {
            let rights = inner.castle_rights(color);
            result.extend(std::iter::repeat((rights.has_kingside()) as u8 as f32).take(8 * 8));
            result.extend(std::iter::repeat((rights.has_queenside()) as u8 as f32).take(8 * 8));
        }

        //en passant
        for square in chess::ALL_SQUARES {
            result.push((inner.en_passant() == Some(square)) as u8 as f32);
        }
    }
}

impl PolicyMapper<ChessBoard> for ChessStdMapper {
    const POLICY_SHAPE: [usize; 3] = [POLICY_CHANNELS, 8, 8];

    fn move_to_index(&self, _: &ChessBoard, mv: ChessMove) -> Option<usize> {
        let classified = ClassifiedMove::from_move(mv);
        let channel = classified.to_channel();
        assert!(channel < POLICY_CHANNELS);

        let from_index = mv.get_source().to_index();
        let index = channel * 8 * 8 + from_index;
        assert!(index < Self::POLICY_SIZE);

        Some(index)
    }

    fn index_to_move(&self, board: &ChessBoard, index: usize) -> Option<ChessMove> {
        let channel = index / (8 * 8);
        let from_index = index % (8 * 8);

        let classified = ClassifiedMove::from_channel(channel);
        let from = index_to_square(from_index);

        classified.to_move(board.inner(), from)
    }
}

fn index_to_square(index: usize) -> Square {
    assert!(index < 8 * 8);
    unsafe { Square::new(index as u8) }
}

#[derive(Debug, Copy, Clone)]
pub enum ClassifiedMove {
    Queen { direction: usize, distance_m1: usize },
    Knight { direction: usize },
    UnderPromotion { direction: usize, piece: usize },
}

impl ClassifiedMove {
    pub fn to_move(self, board: &Board, from: Square) -> Option<ChessMove> {
        match self {
            ClassifiedMove::Queen { direction, distance_m1 } => {
                let (rank_dir, file_dir) = QUEEN_DIRECTIONS[direction];
                let distance = distance_m1 + 1;
                let to = square(
                    from.get_rank().to_index() as isize + distance as isize * rank_dir,
                    from.get_file().to_index() as isize + distance as isize * file_dir,
                )?;

                let moving_pawn = board.piece_on(from) == Some(Piece::Pawn);
                let to_backrank = to.get_rank() == board.side_to_move().to_their_backrank();
                let promotion = if moving_pawn && to_backrank {
                    Some(Piece::Queen)
                } else {
                    None
                };

                Some(ChessMove::new(from, to, promotion))
            }
            ClassifiedMove::Knight { direction } => {
                let (rank_delta, file_delta) = KNIGHT_DELTAS[direction];
                let to = square(
                    from.get_rank().to_index() as isize + rank_delta,
                    from.get_file().to_index() as isize + file_delta,
                )?;

                Some(ChessMove::new(from, to, None))
            }
            ClassifiedMove::UnderPromotion { direction, piece } => {
                let to = square(
                    board.side_to_move().to_their_backrank().to_index() as isize,
                    from.get_file() as isize + (direction as isize - 1),
                )?;

                let promotion = UNDERPROMOTION_PIECES[piece];

                Some(ChessMove::new(from, to, Some(promotion)))
            }
        }
    }

    pub fn from_move(mv: ChessMove) -> Self {
        let from = mv.get_source();
        let to = mv.get_dest();

        let rank_delta = (to.get_rank().to_index() as isize) - (from.get_rank().to_index() as isize);
        let file_delta = (to.get_file().to_index() as isize) - (from.get_file().to_index() as isize);

        // underpromotion
        if let Some(piece) = mv.get_promotion() {
            if let Some(piece) = UNDERPROMOTION_PIECES.iter().index_of(&piece) {
                let direction = (file_delta.signum() + 1) as usize;
                return ClassifiedMove::UnderPromotion { direction, piece };
            }
        }

        // queen
        if let Some(direction) = QUEEN_DIRECTIONS.iter().index_of(&(rank_delta.signum(), file_delta.signum())) {
            let distance = max(rank_delta.abs(), file_delta.abs());

            let (rank_dir, file_dir) = QUEEN_DIRECTIONS[direction];
            if rank_delta == rank_dir * distance && file_delta == file_dir * distance {
                let distance_m1 = (distance - 1) as usize;
                return ClassifiedMove::Queen { direction, distance_m1 };
            }
        }

        // knight
        if let Some(direction) = KNIGHT_DELTAS.iter().index_of(&(rank_delta, file_delta)) {
            return ClassifiedMove::Knight { direction };
        }

        panic!("Could not find move type for {}", mv);
    }

    pub fn to_channel(self) -> usize {
        match self {
            ClassifiedMove::Queen { direction, distance_m1 } => {
                assert!(direction < 8 && distance_m1 < 7);
                direction * 7 + distance_m1
            }
            ClassifiedMove::Knight { direction } => {
                assert!(direction < 8);
                QUEEN_CHANNELS + direction
            }
            ClassifiedMove::UnderPromotion { direction, piece } => {
                assert!(direction < 3 && piece < 3);
                QUEEN_CHANNELS + KNIGHT_CHANNELS + direction * 3 + piece
            }
        }
    }

    pub fn from_channel(channel: usize) -> Self {
        assert!(channel < POLICY_CHANNELS);

        if channel < QUEEN_CHANNELS {
            let direction = channel / 7;
            let distance_m1 = channel % 7;
            ClassifiedMove::Queen { direction, distance_m1 }
        } else if channel < QUEEN_CHANNELS + KNIGHT_CHANNELS {
            let direction = channel - QUEEN_CHANNELS;
            ClassifiedMove::Knight { direction }
        } else {
            let left = channel - (QUEEN_CHANNELS + KNIGHT_CHANNELS);
            assert!(left < UNDERPROMOTION_CHANNELS);
            let direction = left / 3;
            let piece = left % 3;
            ClassifiedMove::UnderPromotion { direction, piece }
        }
    }
}

fn square(rank: isize, file: isize) -> Option<Square> {
    if (0..8).contains(&rank) && (0..8).contains(&file) {
        Some(Square::make_square(Rank::from_index(rank as usize), File::from_index(file as usize)))
    } else {
        None
    }
}

const QUEEN_DISTANCE_COUNT: usize = 7;
const QUEEN_DIRECTION_COUNT: usize = 8;
const KNIGHT_DIRECTION_COUNT: usize = 8;

const QUEEN_CHANNELS: usize = QUEEN_DISTANCE_COUNT * QUEEN_DIRECTION_COUNT;
const KNIGHT_CHANNELS: usize = KNIGHT_DIRECTION_COUNT;
const UNDERPROMOTION_CHANNELS: usize = 3 * 3;

const POLICY_CHANNELS: usize = QUEEN_CHANNELS + KNIGHT_CHANNELS + UNDERPROMOTION_CHANNELS;

// clockwise starting from NNE
const KNIGHT_DELTAS: [(isize, isize); KNIGHT_DIRECTION_COUNT] =
    [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)];

// clockwise starting from N
const QUEEN_DIRECTIONS: [(isize, isize); QUEEN_DIRECTION_COUNT] =
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

const UNDERPROMOTION_PIECES: [Piece; 3] =
    [Piece::Rook, Piece::Bishop, Piece::Knight];
