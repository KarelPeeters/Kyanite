use std::cmp::max;
use std::fmt::Debug;

use board_game::games::chess::ChessBoard;
use chess::{ALL_FILES, ALL_RANKS, ChessMove, Color, File, Piece, Rank, Square};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;
use crate::util::IndexOf;

//TODO try different (policy) embeddings discussed in Discord
#[derive(Debug, Copy, Clone)]
pub struct ChessStdMapper;

impl InputMapper<ChessBoard> for ChessStdMapper {
    const INPUT_BOARD_SIZE: usize = 8;
    // pieces, en passant
    const INPUT_BOOL_PLANES: usize = (2 * 6) + 1;
    // side to move, 50 move counter, repetition counter, castling rights,
    const INPUT_SCALAR_COUNT: usize = 2 + (1 + 1) + (2 * 2);

    fn encode(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &ChessBoard) {
        let inner = board.inner();
        let pov_colors = [inner.side_to_move(), !inner.side_to_move()];
        let pov_ranks = if inner.side_to_move() == Color::White { &ALL_RANKS } else { &ALL_RANKS_REV };

        //TODO maybe remove this? is the game indeed fully symmetric after the pov stuff below?
        //  leave for now but remove once we can reproduce LC0
        //absolute reference for the current player, everything else is from POV
        for color in chess::ALL_COLORS {
            scalars.push((inner.side_to_move() == color) as u8 as f32);
        }

        //castling rights
        for &color in &pov_colors {
            let rights = inner.castle_rights(color);
            scalars.push(rights.has_kingside() as u8 as f32);
            scalars.push(rights.has_queenside() as u8 as f32);
        }

        // counters
        scalars.push(board.repetitions as f32);
        scalars.push(board.non_pawn_or_capture_moves as f32);

        //pieces
        for &color in &pov_colors {
            for piece in chess::ALL_PIECES {
                let color_piece = inner.color_combined(color) & inner.pieces(piece);
                bools.push_block(color_piece.0);
            }
        }

        //en passant
        for &rank in pov_ranks {
            for file in ALL_FILES {
                let square = Square::make_square(rank, file);
                bools.push(inner.en_passant() == Some(square));
            }
        }
    }
}

impl PolicyMapper<ChessBoard> for ChessStdMapper {
    const POLICY_BOARD_SIZE: usize = 8;
    const POLICY_PLANES: usize = POLICY_CHANNELS;

    fn move_to_index(&self, board: &ChessBoard, mv_abs: ChessMove) -> Option<usize> {
        let mv = move_pov(board.inner().side_to_move(), mv_abs);

        let classified = ClassifiedPovMove::from_move(mv);
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

        let classified = ClassifiedPovMove::from_channel(channel);
        let from = square_from_index(from_index);

        let pov = board.inner().side_to_move();
        let from_abs = square_pov(pov, from);
        let moving_pawn = board.inner().piece_on(from_abs) == Some(Piece::Pawn);

        let mv = classified.to_move(moving_pawn, from);
        let mv_abs = mv.map(|mv_pov| move_pov(pov, mv_pov));

        mv_abs
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ClassifiedPovMove {
    Queen { direction: usize, distance_m1: usize },
    Knight { direction: usize },
    UnderPromotion { direction: usize, piece: usize },
}

impl ClassifiedPovMove {
    pub fn to_move(self, moving_pawn: bool, from: Square) -> Option<ChessMove> {
        match self {
            ClassifiedPovMove::Queen { direction, distance_m1 } => {
                let (rank_dir, file_dir) = QUEEN_DIRECTIONS[direction];
                let distance = distance_m1 + 1;
                let to = square(
                    from.get_rank().to_index() as isize + distance as isize * rank_dir,
                    from.get_file().to_index() as isize + distance as isize * file_dir,
                )?;

                let to_backrank = to.get_rank() == Rank::Eighth;
                let promotion = if moving_pawn && to_backrank {
                    Some(Piece::Queen)
                } else {
                    None
                };

                Some(ChessMove::new(from, to, promotion))
            }
            ClassifiedPovMove::Knight { direction } => {
                let (rank_delta, file_delta) = KNIGHT_DELTAS[direction];
                let to = square(
                    from.get_rank().to_index() as isize + rank_delta,
                    from.get_file().to_index() as isize + file_delta,
                )?;

                Some(ChessMove::new(from, to, None))
            }
            ClassifiedPovMove::UnderPromotion { direction, piece } => {
                let to = square(
                    Rank::Eighth.to_index() as isize,
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
                return ClassifiedPovMove::UnderPromotion { direction, piece };
            }
        }

        // queen
        if let Some(direction) = QUEEN_DIRECTIONS.iter().index_of(&(rank_delta.signum(), file_delta.signum())) {
            let distance = max(rank_delta.abs(), file_delta.abs());

            let (rank_dir, file_dir) = QUEEN_DIRECTIONS[direction];
            if rank_delta == rank_dir * distance && file_delta == file_dir * distance {
                let distance_m1 = (distance - 1) as usize;
                return ClassifiedPovMove::Queen { direction, distance_m1 };
            }
        }

        // knight
        if let Some(direction) = KNIGHT_DELTAS.iter().index_of(&(rank_delta, file_delta)) {
            return ClassifiedPovMove::Knight { direction };
        }

        panic!("Could not find move type for {}", mv);
    }

    pub fn to_channel(self) -> usize {
        match self {
            ClassifiedPovMove::Queen { direction, distance_m1 } => {
                assert!(direction < 8 && distance_m1 < 7);
                direction * 7 + distance_m1
            }
            ClassifiedPovMove::Knight { direction } => {
                assert!(direction < 8);
                QUEEN_CHANNELS + direction
            }
            ClassifiedPovMove::UnderPromotion { direction, piece } => {
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
            ClassifiedPovMove::Queen { direction, distance_m1 }
        } else if channel < QUEEN_CHANNELS + KNIGHT_CHANNELS {
            let direction = channel - QUEEN_CHANNELS;
            ClassifiedPovMove::Knight { direction }
        } else {
            let left = channel - (QUEEN_CHANNELS + KNIGHT_CHANNELS);
            assert!(left < UNDERPROMOTION_CHANNELS);
            let direction = left / 3;
            let piece = left % 3;
            ClassifiedPovMove::UnderPromotion { direction, piece }
        }
    }
}

fn square_from_index(index: usize) -> Square {
    assert!(index < 8 * 8);
    Square::make_square(
        Rank::from_index(index / 8),
        File::from_index(index % 8),
    )
}

fn square(rank: isize, file: isize) -> Option<Square> {
    if (0..8).contains(&rank) && (0..8).contains(&file) {
        Some(Square::make_square(Rank::from_index(rank as usize), File::from_index(file as usize)))
    } else {
        None
    }
}

/// View a square from the given pov.
/// This function can be used for both the abs->pov and pov->abs directions.
fn square_pov(pov: Color, sq: Square) -> Square {
    match pov {
        Color::White => sq,
        Color::Black => {
            let rank_pov = Rank::from_index(7 - sq.get_rank().to_index());
            Square::make_square(rank_pov, sq.get_file())
        }
    }
}

fn move_pov(pov: Color, mv: ChessMove) -> ChessMove {
    ChessMove::new(
        square_pov(pov, mv.get_source()),
        square_pov(pov, mv.get_dest()),
        mv.get_promotion(),
    )
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

const ALL_RANKS_REV: [Rank; 8] = [
    Rank::Eighth,
    Rank::Seventh,
    Rank::Sixth,
    Rank::Fifth,
    Rank::Fourth,
    Rank::Third,
    Rank::Second,
    Rank::First,
];