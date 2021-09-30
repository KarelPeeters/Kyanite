use std::cmp::{max, min};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::Board;
use board_game::wdl::POV;
use internal_iterator::InternalIterator;
use itertools::zip;
use safe_transmute::transmute_to_bytes;
use serde::Serialize;

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::BoardMapper;
use crate::selfplay::simulation::Simulation;

#[derive(Serialize)]
struct MetaData<'a> {
    game: &'a str,

    scalar_count: usize,
    board_bool_planes: usize,
    board_scalar_count: usize,
    policy_planes: usize,

    game_count: usize,
    position_count: usize,

    max_game_length: i32,
    min_game_length: i32,

    position_offsets_offset: u64,
}

#[derive(Debug)]
pub struct BinaryOutput<B: Board, M: BoardMapper<B>> {
    game: String,

    bin_write: BufWriter<File>,
    path: PathBuf,

    game_count: usize,
    max_game_length: Option<i32>,
    min_game_length: Option<i32>,

    position_offsets: Vec<u64>,
    finished: bool,

    mapper: M,
    ph: PhantomData<B>,
}

type Result<T> = std::io::Result<T>;

impl<B: Board, M: BoardMapper<B>> BinaryOutput<B, M> {
    pub fn new(path: impl AsRef<Path>, game: &str, mapper: M) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        assert!(path.extension().is_none());

        let path_bin = path.with_extension("bin");
        let bin_write = BufWriter::new(File::create(path_bin)?);

        Ok(BinaryOutput {
            game: game.to_string(),
            bin_write,
            path,

            game_count: 0,
            max_game_length: None,
            min_game_length: None,

            position_offsets: vec![],

            finished: false,
            mapper,
            ph: PhantomData,
        })
    }

    const SCALAR_COUNT: usize = 5 + 2 + 3 * 3;

    pub fn append(&mut self, sim: Simulation<B>) -> Result<()> {
        let game_id = self.game_count;
        self.game_count += 1;

        let game_length = sim.positions.len();

        self.max_game_length = Some(max(game_length as i32, self.max_game_length.unwrap_or(-1)));
        self.min_game_length = Some(min(game_length as i32, self.min_game_length.unwrap_or(i32::MAX)));

        let mut scalars: Vec<f32> = vec![];
        let mut board_bools = BitBuffer::new(M::INPUT_BOOL_COUNT);
        let mut board_scalars: Vec<f32> = vec![];
        let mut policy_indices: Vec<u32> = vec![];

        for (pos_index, pos) in sim.positions.iter().enumerate() {
            let board = &pos.board;
            let player = board.next_player();

            // board
            self.mapper.encode(&mut board_bools, &mut board_scalars, board);

            assert_eq!(M::INPUT_BOOL_COUNT, board_bools.len());
            assert_eq!(M::INPUT_SCALAR_COUNT, board_scalars.len());
            assert_eq!((M::INPUT_BOOL_COUNT + 7) / 8, board_bools.storage().len());

            // policy
            let mut got_none = false;
            let mut available_mv_count = 0;
            board.available_moves().for_each(|mv: B::Move| {
                available_mv_count += 1;
                match self.mapper.move_to_index(board, mv) {
                    Some(index) => {
                        policy_indices.push(index as u32);
                    }
                    None => {
                        got_none = true
                    }
                }
            });

            assert!(!got_none || available_mv_count == 1);
            assert_eq!(available_mv_count, pos.zero_evaluation.policy.len());

            // scalar float data
            scalars.push(game_id as f32);
            scalars.push(pos_index as f32);
            scalars.push(game_length as f32);
            scalars.push(pos.zero_visits as f32);
            scalars.push(available_mv_count as f32);

            scalars.push(kdl_divergence(&pos.zero_evaluation.wdl.to_slice(), &pos.net_evaluation.wdl.to_slice()));
            scalars.push(kdl_divergence(&pos.zero_evaluation.policy, &pos.net_evaluation.policy));

            scalars.extend_from_slice(&sim.outcome.pov(player).to_wdl().to_slice());
            scalars.extend_from_slice(&pos.zero_evaluation.wdl.to_slice());
            scalars.extend_from_slice(&pos.net_evaluation.wdl.to_slice());
            assert_eq!(Self::SCALAR_COUNT, scalars.len());

            // save current offset
            let curr_offset = self.bin_write.seek(SeekFrom::Current(0))?;
            self.position_offsets.push(curr_offset);

            // actually write stuff to the bin file
            self.bin_write.write_all(transmute_to_bytes(&scalars))?;
            self.bin_write.write_all(transmute_to_bytes(board_bools.storage()))?;
            self.bin_write.write_all(transmute_to_bytes(&board_scalars))?;
            self.bin_write.write_all(transmute_to_bytes(&policy_indices))?;
            self.bin_write.write_all(transmute_to_bytes(&pos.zero_evaluation.policy))?;

            scalars.clear();
            board_bools.clear();
            board_scalars.clear();
            policy_indices.clear();
        }

        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            panic!("This output is already finished")
        }
        self.finished = true;

        let position_offsets_offset = self.bin_write.seek(SeekFrom::Current(0))?;
        self.bin_write.write_all(transmute_to_bytes(&self.position_offsets))?;
        self.bin_write.flush()?;

        let meta = MetaData {
            game: &self.game,
            scalar_count: Self::SCALAR_COUNT,
            board_bool_planes: M::INPUT_BOOL_PLANES,
            board_scalar_count: M::INPUT_SCALAR_COUNT,
            policy_planes: M::POLICY_PLANES,
            game_count: self.game_count,
            position_count: self.position_offsets.len(),
            max_game_length: self.max_game_length.unwrap_or(-1),
            min_game_length: self.min_game_length.unwrap_or(-1),
            position_offsets_offset,
        };

        let path_json_tmp = self.path.with_extension("json.tmp");
        let path_json = self.path.with_extension("json");

        let mut json_writer = BufWriter::new(File::create(&path_json_tmp)?);
        serde_json::to_writer_pretty(&mut json_writer, &meta)?;
        json_writer.flush()?;
        drop(json_writer);

        std::fs::rename(path_json_tmp, path_json)?;

        Ok(())
    }

    pub fn game_count(&self) -> usize {
        self.game_count
    }
}

/// Calculates `D_KDL(P || Q)`, read as _the divergence of P from Q_,
/// a measure of how different two probability distributions are.
/// See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
fn kdl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len());

    zip(p, q)
        .map(|(&p, &q)| p * (p / q).ln())
        .sum()
}

#[cfg(test)]
mod test {
    use crate::mapping::binary_output::kdl_divergence;

    #[test]
    fn basic_kdl() {
        let p = [9.0 / 25.0, 12.0 / 25.0, 4.0 / 25.0];
        let q = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let kdl = kdl_divergence(&p, &q);
        let expected = 0.0852996;
        assert!((kdl - expected).abs() < 1e-5);
    }
}