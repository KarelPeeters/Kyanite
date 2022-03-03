use std::cmp::{max, min};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::Board;
use board_game::wdl::{POV, WDL};
use internal_iterator::InternalIterator;
use safe_transmute::transmute_to_bytes;
use serde::Serialize;

use kz_core::mapping::bit_buffer::BitBuffer;
use kz_core::mapping::BoardMapper;
use kz_core::zero::node::ZeroValues;
use kz_util::kdl_divergence;

use crate::simulation::{Position, Simulation};

#[derive(Serialize)]
struct MetaData<'a> {
    game: &'a str,

    input_bool_shape: &'a [usize],
    input_scalar_count: usize,
    policy_shape: &'a [usize],

    game_count: usize,
    position_count: usize,

    max_game_length: i32,
    min_game_length: i32,
    root_wdl: [f32; 3],

    scalar_names: &'static [&'static str],
}

#[derive(Debug)]
pub struct BinaryOutput<B: Board, M: BoardMapper<B>> {
    game: String,
    path: PathBuf,

    bin_write: BufWriter<File>,
    off_write: BufWriter<File>,
    json_tmp_write: BufWriter<File>,

    game_count: usize,
    position_count: usize,

    max_game_length: Option<i32>,
    min_game_length: Option<i32>,
    total_root_wdl: WDL<f32>,

    next_offset: u64,
    finished: bool,

    mapper: M,
    ph: PhantomData<B>,
}

type Result<T> = std::io::Result<T>;

const SCALAR_NAMES: &[&str] = &[
    "game_id", "pos_index", "game_length", "zero_visits", "available_mv_count", "played_mv",
    "kdl_policy",
    "final_v", "final_wdl_w", "final_wdl_d", "final_wdl_l", "final_moves_left",
    "zero_v", "zero_wdl_w", "zero_wdl_d", "zero_wdl_l", "zero_moves_left",
    "net_v", "net_wdl_w", "net_wdl_d", "net_wdl_l", "net_moves_left",
];

impl<B: Board, M: BoardMapper<B>> BinaryOutput<B, M> {
    pub fn new(path: impl AsRef<Path>, game: &str, mapper: M) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        assert!(path.extension().is_none(), "Binary output path should not have an extension, .bin and .json are added automatically");

        //TODO try buffer sizes again
        let bin_write = BufWriter::new(File::create(path.with_extension("bin"))?);
        let off_write = BufWriter::new(File::create(path.with_extension("off"))?);
        let json_tmp_write = BufWriter::new(File::create(path.with_extension("json.tmp"))?);

        Ok(BinaryOutput {
            game: game.to_string(),

            bin_write,
            off_write,
            json_tmp_write,

            path,

            game_count: 0,
            position_count: 0,

            max_game_length: None,
            min_game_length: None,
            total_root_wdl: WDL::default(),

            next_offset: 0,

            finished: false,
            mapper,
            ph: PhantomData,
        })
    }

    pub fn append(&mut self, simulation: &Simulation<B>) -> Result<()> {
        let Simulation { outcome, positions } = simulation;
        assert!(!positions.is_empty(), "Simulation cannot be empty");

        // collect metadata statistics
        let game_id = self.game_count;
        let game_length = positions.len();

        self.game_count += 1;
        self.position_count += game_length;

        self.max_game_length = Some(max(game_length as i32, self.max_game_length.unwrap_or(-1)));
        self.min_game_length = Some(min(game_length as i32, self.min_game_length.unwrap_or(i32::MAX)));
        self.total_root_wdl += outcome.pov(positions[0].board.next_player()).to_wdl();

        let mut scalars: Vec<f32> = vec![];
        let mut board_bools = BitBuffer::new(self.mapper.input_bool_len());
        let mut board_scalars: Vec<f32> = vec![];
        let mut policy_indices: Vec<u32> = vec![];

        for (pos_index, position) in positions.iter().enumerate() {
            let &Position {
                ref board, should_store, played_mv, zero_visits,
                ref zero_evaluation, ref net_evaluation
            } = position;

            if !should_store { continue; }

            let player = board.next_player();
            let moves_left = (positions.len() - pos_index) as f32;
            let final_values = ZeroValues::from_outcome(outcome.pov(player), moves_left);

            // board
            self.mapper.encode(&mut board_bools, &mut board_scalars, board);

            assert_eq!(self.mapper.input_bool_len(), board_bools.len());
            assert_eq!(self.mapper.input_scalar_count(), board_scalars.len());
            assert_eq!((self.mapper.input_bool_len() + 7) / 8, board_bools.storage().len());

            // policy
            //TODO get rid of this "forced pass" concept and just map it to a separate move index
            let mut forced_pass = false;
            let mut available_mv_count = 0;
            board.available_moves().for_each(|mv: B::Move| {
                available_mv_count += 1;
                match self.mapper.move_to_index(board, mv) {
                    Some(index) => {
                        policy_indices.push(index as u32);
                    }
                    None => {
                        forced_pass = true
                    }
                }
            });
            let played_mv_index = self.mapper.move_to_index(board, played_mv);

            // check that everything makes sense
            assert!(!forced_pass || available_mv_count == 1);
            assert_eq!(available_mv_count, zero_evaluation.policy.len());
            assert_eq!(available_mv_count, net_evaluation.policy.len());
            zero_evaluation.assert_normalized_or_nan();
            net_evaluation.assert_normalized_or_nan();

            // scalar float data
            scalars.push(game_id as f32);
            scalars.push(pos_index as f32);
            scalars.push(game_length as f32);
            scalars.push(zero_visits as f32);
            scalars.push(if forced_pass { 0 } else { available_mv_count } as f32);
            scalars.push(played_mv_index.map_or(-1.0, |i| i as f32));

            scalars.push(kdl_divergence(&zero_evaluation.policy, &net_evaluation.policy));

            push_values(&mut scalars, final_values);
            push_values(&mut scalars, zero_evaluation.values);
            push_values(&mut scalars, net_evaluation.values);

            assert_eq!(SCALAR_NAMES.len(), scalars.len());

            // save current offset
            // we keep track of the offset ourselves because seeking/stream_position flushes the buffer and is slow
            debug_assert_eq!(self.next_offset, self.bin_write.stream_position()?);
            self.off_write.write_all(&self.next_offset.to_le_bytes())?;

            // actually write stuff to the bin file
            let data_to_write = [
                transmute_to_bytes(&scalars),
                transmute_to_bytes(board_bools.storage()),
                transmute_to_bytes(&board_scalars),
                transmute_to_bytes(&policy_indices),
                transmute_to_bytes(if !forced_pass { &zero_evaluation.policy } else { &[] }),
            ];
            for data in data_to_write {
                self.bin_write.write_all(data)?;
                self.next_offset += data.len() as u64;
            }

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

        let meta = MetaData {
            game: &self.game,
            scalar_names: SCALAR_NAMES,
            input_bool_shape: &self.mapper.input_bool_shape(),
            input_scalar_count: self.mapper.input_scalar_count(),
            policy_shape: self.mapper.policy_shape(),
            game_count: self.game_count,
            position_count: self.position_count,
            max_game_length: self.max_game_length.unwrap_or(-1),
            min_game_length: self.min_game_length.unwrap_or(-1),
            root_wdl: (self.total_root_wdl / self.game_count as f32).to_slice(),
        };

        serde_json::to_writer_pretty(&mut self.json_tmp_write, &meta)?;
        self.json_tmp_write.flush()?;
        self.bin_write.flush()?;
        self.off_write.flush()?;

        let path_json_tmp = self.path.with_extension("json.tmp");
        let path_json = self.path.with_extension("json");
        std::fs::rename(path_json_tmp, path_json)?;

        Ok(())
    }

    pub fn game_count(&self) -> usize {
        self.game_count
    }
}

fn push_values(scalars: &mut Vec<f32>, values: ZeroValues) {
    let ZeroValues { value, wdl, moves_left } = values;

    scalars.push(value);
    scalars.extend_from_slice(&wdl.to_slice());
    scalars.push(moves_left);
}
