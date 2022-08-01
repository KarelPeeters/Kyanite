use board_game::arimaa_engine_step::{Action, Direction, Piece, PushPullState, Square};
use board_game::board::{Board, Player};
use board_game::games::arimaa::ArimaaBoard;
use board_game::util::bitboard::BitBoard8;

use kz_util::sequence::IndexOf;

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::{InputMapper, MuZeroMapper, PolicyMapper};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ArimaaSplitMapper;

impl InputMapper<ArimaaBoard> for ArimaaSplitMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [4 * 6 + 2, 8, 8]
    }

    fn input_scalar_count(&self) -> usize {
        12
    }

    fn encode_input(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &ArimaaBoard) {
        let state = board.state();
        let next_player = board.next_player();

        let pp_state = state.as_play_phase().map(|p| p.push_pull_state());

        let (place, play, pull, push) = match pp_state {
            None => (true, false, None, None),
            Some(PushPullState::None) => (false, true, None, None),
            Some(PushPullState::PossiblePull(square, piece)) => (false, true, Some((square, piece)), None),
            Some(PushPullState::MustCompletePush(square, piece)) => (false, true, None, Some((square, piece))),
        };

        // main pieces
        for player in [next_player, next_player.other()] {
            for piece in Piece::ALL {
                let board_raw = board.bits_for_piece(piece, player);
                let board_pov = board_from_pov(board_raw, next_player);
                bools.push_block(board_pov.0)
            }
        }

        // pull and push squares
        append_push_pull_planes(bools, pull, next_player);
        append_push_pull_planes(bools, push, next_player);

        // placement and traps
        // TODO fix inconsistent placement order in board
        //   gold places middle first then edge, silver the other way around
        bools.push_block(board_from_pov(board.placement(), next_player).0);
        bools.push_block(board_from_pov(ArimaaBoard::TRAP_MASK, next_player).0);

        // main scalars
        scalars.push(place as u8 as f32);
        scalars.push(play as u8 as f32);
        scalars.push(pull.is_some() as u8 as f32);
        scalars.push(push.is_some() as u8 as f32);

        for i in 0..ArimaaBoard::MAX_STEPS_PER_TURN {
            scalars.push((i == board.steps_taken()) as u8 as f32);
        }

        // metadata
        scalars.push((board.next_player() == Player::A) as u8 as f32);
        scalars.push((board.next_player() == Player::B) as u8 as f32);
        scalars.push(board.history_len() as f32);
        scalars.push(state.move_number() as f32);
    }
}

fn append_push_pull_planes(bools: &mut BitBuffer, pair: Option<(Square, Piece)>, pov: Player) {
    let (board_raw, piece) = match pair {
        None => (BitBoard8::EMPTY, None),
        Some((square, piece)) => (BitBoard8(square.as_bit_board()), Some(piece)),
    };

    let board_pov = board_from_pov(board_raw, pov);

    for curr in Piece::ALL {
        if piece == Some(curr) {
            bools.push_block(board_pov.0)
        } else {
            bools.push_block(0);
        }
    }
}

impl PolicyMapper<ArimaaBoard> for ArimaaSplitMapper {
    fn policy_shape(&self) -> &[usize] {
        &[1 + 6 + 4 * 8 * 8]
    }

    fn move_to_index(&self, board: &ArimaaBoard, mv: Action) -> usize {
        match mv {
            Action::Pass => 0,
            Action::Place(piece) => {
                let piece_index = Piece::ALL.iter().index_of(&piece).unwrap();
                1 + piece_index
            }
            Action::Move(square_abs, direction_abs) => {
                let pov = board.next_player();
                let square_pov = square_from_pov(square_abs, pov);
                let direction_pov = direction_from_pov(direction_abs, pov);

                let direction_index = Direction::ALL.iter().index_of(&direction_pov).unwrap();
                let tensor_index = direction_index * 64 + square_pov.index();
                1 + Piece::ALL.len() + tensor_index
            }
        }
    }

    fn index_to_move(&self, board: &ArimaaBoard, index: usize) -> Option<Action> {
        let piece_count = Piece::ALL.len();

        let mv = if index == 0 {
            Action::Pass
        } else if index < 1 + piece_count {
            Action::Place(Piece::ALL[index - 1])
        } else {
            let tensor_index = index - 1 - piece_count;

            let square_pov = Square::from_index((tensor_index % 64) as u8);
            let direction_pov = Direction::ALL[tensor_index / 64];

            let pov = board.next_player();
            let square_abs = square_from_pov(square_pov, pov);
            let direction_abs = direction_from_pov(direction_pov, pov);

            Action::Move(square_abs, direction_abs)
        };

        Some(mv)
    }
}

impl MuZeroMapper<ArimaaBoard> for ArimaaSplitMapper {
    fn state_board_size(&self) -> usize {
        todo!()
    }

    fn encoded_move_shape(&self) -> [usize; 3] {
        todo!()
    }

    fn encode_mv(&self, _: &mut Vec<f32>, _: usize) {
        todo!()
    }
}

fn board_from_pov(board: BitBoard8, player: Player) -> BitBoard8 {
    match player {
        Player::A => board,
        Player::B => board.flip_y(),
    }
}

fn square_from_pov(square: Square, player: Player) -> Square {
    match player {
        Player::A => square,
        Player::B => Square::from_bit_board(BitBoard8(square.as_bit_board()).flip_y().0),
    }
}

fn direction_from_pov(direction: Direction, player: Player) -> Direction {
    match (player, direction) {
        (Player::A, direction) => direction,
        (Player::B, Direction::Up) => Direction::Down,
        (Player::B, Direction::Down) => Direction::Up,
        (Player::B, Direction::Left) => Direction::Left,
        (Player::B, Direction::Right) => Direction::Right,
    }
}
