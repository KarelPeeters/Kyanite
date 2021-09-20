use std::io::Write;
use std::marker::PhantomData;

use board_game::board::Board;
use internal_iterator::InternalIterator;
use safe_transmute::transmute_to_bytes;

use crate::mapping::BoardMapper;
use crate::selfplay::core::{Output, Simulation};

#[derive(Debug)]
pub struct BinaryOutput<W: Write, B: Board, M: BoardMapper<B>> {
    mapper: M,
    writer: W,
    next_game_id: usize,
    ph: PhantomData<B>,
}

impl<W: Write, B: Board, M: BoardMapper<B>> BinaryOutput<W, B, M> {
    pub fn new(mapper: M, writer: W) -> Self {
        BinaryOutput { mapper, writer, next_game_id: 0, ph: PhantomData }
    }
}

impl<W: Write, B: Board, M: BoardMapper<B>> Output<B> for BinaryOutput<W, B, M> {
    fn append(&mut self, simulation: Simulation<B>) {
        let game_id = self.next_game_id;
        self.next_game_id += 1;

        let mut data = vec![];
        let position_count = simulation.positions.len();

        // fill data
        for pos in simulation.iter() {
            let board = pos.board;

            let moves: Vec<_> = board.available_moves().collect();
            let policy = pos.evaluation.policy;

            // game id
            data.push(game_id as f32);

            // wdls
            data.extend_from_slice(&pos.final_wdl.to_slice());
            data.extend_from_slice(&pos.evaluation.wdl.to_slice());

            // policy mask
            for i in 0..M::POLICY_SIZE {
                let is_available = self.mapper.index_to_move(&board, i)
                    .map_or(0.0, |mv| board.is_available_move(mv) as u8 as f32);
                data.push(is_available);
            }

            // policy
            for i in 0..M::POLICY_SIZE {
                // get the policy value if the move exists and is available
                let p = self.mapper.index_to_move(&board, i)
                    .filter(|&mv| board.is_available_move(mv))
                    .map_or(0.0, |mv| {
                        let policy_index = moves.iter().position(|&cand| cand == mv).unwrap();
                        policy[policy_index]
                    });
                data.push(p);
            }

            // board input
            let start = data.len();
            self.mapper.append_board_to(&mut data, &board);
            assert_eq!(M::INPUT_SIZE, data.len() - start);
        }

        let expected_single_size = 1 + 2 * 3 + 2 * M::POLICY_SIZE + M::INPUT_SIZE;
        let expected_data_size = position_count * expected_single_size;
        assert_eq!(expected_data_size, data.len());

        // write data
        self.writer.write_all(transmute_to_bytes(&data))
            .expect("Failed to write data");
        self.writer.flush()
            .expect("Failed to flush");
    }
}

