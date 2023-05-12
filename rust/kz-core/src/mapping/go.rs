use board_game::board::{Board, Player};
use board_game::games::go::{FlatTile, GoBoard, Move, State, Tile};

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::{InputMapper, MuZeroMapper, PolicyMapper};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct GoStdMapper {
    max_size: u8,
    max_area: u16,
    policy_shape: [usize; 1],
}

impl GoStdMapper {
    pub fn new(max_size: u8) -> Self {
        let max_area = max_size as u16 * max_size as u16;
        GoStdMapper {
            max_size,
            max_area,
            policy_shape: [1 + max_area as usize],
        }
    }

    fn move_to_index(&self, mv: Move) -> usize {
        match mv {
            Move::Pass => 0,
            Move::Place(tile) => 1 + tile.to_flat(self.max_size).index() as usize,
        }
    }

    fn index_to_move(&self, index: usize) -> Move {
        match index {
            0 => Move::Pass,
            _ => {
                let tile_index = index - 1;
                assert!(tile_index < self.max_area as usize);
                Move::Place(FlatTile::new(tile_index as u16).to_tile(self.max_size))
            }
        }
    }
}

impl InputMapper<GoBoard> for GoStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        // stones_us, stones_them, in-board, illegal_move (ko)
        let channels = 2 + 1 + 1;
        [channels, self.max_size as usize, self.max_size as usize]
    }

    fn input_scalar_count(&self) -> usize {
        // state: black_turn, white_turn, pass_count_1, pass_count_2
        // settings: komi
        // rules: multi_suicide
        6
    }

    fn encode_input(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &GoBoard) {
        assert!(board.size() <= self.max_size);
        let next = board.next_player();
        let size = board.size();

        // bools
        for color in [next, next.other()] {
            for tile in Tile::all(self.max_size) {
                bools.push(tile.exists(size) && board.stone_at(tile) == Some(color));
            }
        }
        for tile in Tile::all(self.max_size) {
            bools.push(tile.exists(size));
        }
        for tile in Tile::all(self.max_size) {
            let exists_empty = tile.exists(size) && board.stone_at(tile).is_none();
            let is_available = board.is_available_move(Move::Place(tile)).unwrap_or(true);
            bools.push(exists_empty && !is_available);
        }

        // scalars
        let komi_pov = match next {
            Player::A => board.komi(),
            Player::B => -board.komi(),
        };
        let (pass_1, pass_2) = match board.state() {
            State::Normal => (false, false),
            State::Passed => (true, false),
            State::Done(_) => (false, true),
        };

        scalars.push((board.next_player() == Player::A) as u8 as f32);
        scalars.push((board.next_player() == Player::B) as u8 as f32);
        scalars.push(pass_1 as u8 as f32);
        scalars.push(pass_2 as u8 as f32);
        scalars.push(komi_pov.as_float() / 15.0);
        scalars.push(board.rules().allow_multi_stone_suicide as u8 as f32);
    }
}

impl PolicyMapper<GoBoard> for GoStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &self.policy_shape
    }

    fn move_to_index(&self, _: &GoBoard, mv: Move) -> usize {
        self.move_to_index(mv)
    }

    fn index_to_move(&self, _: &GoBoard, index: usize) -> Option<Move> {
        Some(self.index_to_move(index))
    }
}

// TODO use better pass encoding that doesn't take up an entire plane
impl MuZeroMapper<GoBoard> for GoStdMapper {
    fn state_board_size(&self) -> usize {
        self.max_size as usize
    }

    fn encoded_move_shape(&self) -> [usize; 3] {
        [2, self.max_size as usize, self.max_size as usize]
    }

    fn encode_mv(&self, result: &mut Vec<f32>, mv_index: usize) {
        let mv = self.index_to_move(mv_index);
        for _ in Tile::all(self.max_size) {
            result.push((mv == Move::Pass) as u8 as f32);
        }
        for tile in Tile::all(self.max_size) {
            result.push((mv == Move::Place(tile)) as u8 as f32);
        }
    }
}
