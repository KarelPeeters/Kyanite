use std::cmp::{max, min};
use std::fs::File;
use std::io;
use std::io::{BufWriter, Seek, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::{Board, Outcome};
use board_game::wdl::{POV, WDL};
use bytemuck::cast_slice;
use internal_iterator::InternalIterator;
use serde::Serialize;

use kz_core::mapping::bit_buffer::BitBuffer;
use kz_core::mapping::BoardMapper;
use kz_core::zero::node::ZeroValues;
use kz_util::math::kdl_divergence;

use crate::simulation::{Position, Simulation};

#[derive(Serialize)]
struct MetaData<'a> {
    game: &'a str,

    input_bool_shape: &'a [usize],
    input_scalar_count: usize,
    policy_shape: &'a [usize],

    game_count: usize,
    position_count: usize,
    includes_terminal_positions: bool,

    max_game_length: i32,
    min_game_length: i32,
    root_wdl: [f32; 3],
    hit_move_limit: f32,

    scalar_names: &'static [&'static str],
}

//TODO include terminal position in output?
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

    total_root_wdl: WDL<u64>,
    hit_move_limit_count: u64,

    next_offset: u64,
    finished: bool,

    mapper: M,
    ph: PhantomData<B>,
}

#[derive(Debug)]
struct Scalars {
    game_id: usize,
    pos_index: usize,
    game_length: usize,
    zero_visits: u64,
    is_full_search: bool,
    is_final_position: bool,
    is_terminal: bool,
    hit_move_limit: bool,
    available_mv_count: usize,
    played_mv: isize,
    kdl_policy: f32,
    final_values: ZeroValues,
    zero_values: ZeroValues,
    net_values: ZeroValues,
}

impl<B: Board, M: BoardMapper<B>> BinaryOutput<B, M> {
    pub fn new(path: impl AsRef<Path>, game: &str, mapper: M) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        assert!(
            path.extension().is_none(),
            "Binary output path should not have an extension, .bin and .json are added automatically"
        );

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
            hit_move_limit_count: 0,

            next_offset: 0,

            finished: false,
            mapper,
            ph: PhantomData,
        })
    }

    pub fn append(&mut self, simulation: &Simulation<B>) -> io::Result<()> {
        let Simulation { positions, final_board } = simulation;
        assert!(!positions.is_empty(), "Simulation cannot be empty");

        // collect metadata statistics
        let game_id = self.game_count;
        let game_length = positions.len();

        self.game_count += 1;
        self.position_count += 1 + game_length;

        self.max_game_length = Some(max(game_length as i32, self.max_game_length.unwrap_or(-1)));
        self.min_game_length = Some(min(game_length as i32, self.min_game_length.unwrap_or(i32::MAX)));

        let outcome = final_board.outcome().unwrap_or(Outcome::Draw);
        self.total_root_wdl += outcome.pov(positions[0].board.next_player()).to_wdl();
        self.hit_move_limit_count += final_board.outcome().is_none() as u8 as u64;

        // write of the positions
        for (pos_index, position) in positions.iter().enumerate() {
            let &Position {
                ref board,
                is_full_search,
                played_mv,
                zero_visits,
                ref zero_evaluation,
                ref net_evaluation,
            } = position;

            let (available_mv_count, forced_pass, policy_indices) = collect_policy_indices(board, self.mapper);
            assert_eq!(available_mv_count, zero_evaluation.policy.len());
            assert_eq!(available_mv_count, net_evaluation.policy.len());
            let used_policy_values: &[f32] = if forced_pass {
                assert_eq!(available_mv_count, 1);
                &[]
            } else {
                &zero_evaluation.policy
            };

            let played_mv_index = self.mapper.move_to_index(board, played_mv);
            let kdl_policy = kdl_divergence(&zero_evaluation.policy, &net_evaluation.policy);
            let moves_left = game_length + 1 - pos_index;

            let scalars = Scalars {
                game_id,
                pos_index,
                game_length,
                zero_visits,
                is_full_search,
                is_final_position: false,
                is_terminal: false,
                hit_move_limit: false,
                available_mv_count: used_policy_values.len(),
                played_mv: played_mv_index.map_or(-1, |mv| mv as isize),
                kdl_policy,
                final_values: ZeroValues::from_outcome(outcome.pov(board.next_player()), moves_left as f32),
                zero_values: zero_evaluation.values,
                net_values: net_evaluation.values,
            };

            self.append_position(board, &scalars, &policy_indices, used_policy_values)?;
        }

        let scalars = Scalars {
            game_id,
            pos_index: game_length,
            game_length,
            zero_visits: 0,
            is_full_search: false,
            is_final_position: true,
            is_terminal: final_board.is_done(),
            hit_move_limit: !final_board.is_done(),
            available_mv_count: 0,
            played_mv: -1,
            kdl_policy: f32::NAN,
            final_values: ZeroValues::from_outcome(outcome.pov(final_board.next_player()), 0.0),
            zero_values: ZeroValues::nan(),
            //TODO in theory we could ask the network, but this is only really meaningful for muzero
            net_values: ZeroValues::nan(),
        };

        self.append_position(&final_board, &scalars, &[], &[])?;

        Ok(())
    }

    fn append_position(
        &mut self,
        board: &B,
        scalars: &Scalars,
        policy_indices: &[u32],
        policy_values: &[f32],
    ) -> io::Result<()> {
        // encode board
        let mut board_bools = BitBuffer::new(self.mapper.input_bool_len());
        let mut board_scalars = vec![];
        self.mapper.encode_input(&mut board_bools, &mut board_scalars, board);

        assert_eq!(self.mapper.input_bool_len(), board_bools.len());
        assert_eq!(self.mapper.input_scalar_count(), board_scalars.len());
        assert_eq!((self.mapper.input_bool_len() + 7) / 8, board_bools.storage().len());

        // check that everything makes sense
        let policy_len = policy_indices.len();
        assert_eq!(policy_len, policy_values.len());
        assert_normalized_or_nan(scalars.zero_values.wdl.sum());
        assert_normalized_or_nan(scalars.net_values.wdl.sum());
        assert_normalized_or_nan(scalars.final_values.wdl.sum());
        if policy_len != 0 {
            assert_normalized(policy_values.iter().sum());
        }

        // save current offset
        // we keep track of the offset ourselves because seeking/stream_position flushes the buffer and is slow
        debug_assert_eq!(self.next_offset, self.bin_write.stream_position()?);
        self.off_write.write_all(&self.next_offset.to_le_bytes())?;

        // actually write stuff to the bin file
        let scalars = scalars.to_vec();
        let data_to_write: &[&[u8]] = &[
            cast_slice(&scalars),
            cast_slice(board_bools.storage()),
            cast_slice(&board_scalars),
            cast_slice(policy_indices),
            cast_slice(policy_values),
        ];
        for &data in data_to_write {
            self.bin_write.write_all(data)?;
            self.next_offset += data.len() as u64;
        }

        Ok(())
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            panic!("This output is already finished")
        }
        self.finished = true;

        let meta = MetaData {
            game: &self.game,
            scalar_names: Scalars::NAMES,
            input_bool_shape: &self.mapper.input_bool_shape(),
            input_scalar_count: self.mapper.input_scalar_count(),
            policy_shape: self.mapper.policy_shape(),
            game_count: self.game_count,
            position_count: self.position_count,
            includes_terminal_positions: true,
            max_game_length: self.max_game_length.unwrap_or(-1),
            min_game_length: self.min_game_length.unwrap_or(-1),
            root_wdl: (self.total_root_wdl.cast::<f32>() / self.game_count as f32).to_slice(),
            hit_move_limit: self.hit_move_limit_count as f32 / self.game_count as f32,
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

fn collect_policy_indices<B: Board, M: BoardMapper<B>>(board: &B, mapper: M) -> (usize, bool, Vec<u32>) {
    //TODO get rid of this "forced pass" concept and just map it to a separate move index
    let mut forced_pass = false;
    let mut policy_indices = vec![];
    let mut available_mv_count = 0;

    board.available_moves().for_each(|mv: B::Move| {
        available_mv_count += 1;
        match mapper.move_to_index(board, mv) {
            Some(index) => {
                assert!(!forced_pass);
                policy_indices.push(index as u32);
            }
            None => forced_pass = true,
        }
    });

    (available_mv_count, forced_pass, policy_indices)
}

fn assert_normalized_or_nan(x: f32) {
    assert!(x.is_nan() || (1.0 - x).abs() < 0.001);
}

fn assert_normalized(x: f32) {
    assert!((1.0 - x).abs() < 0.001);
}

impl Scalars {
    const NAMES: &'static [&'static str] = &[
        "game_id",
        "pos_index",
        "game_length",
        "zero_visits",
        "is_full_search",
        "is_final_position",
        "is_terminal",
        "hit_move_limit",
        "available_mv_count",
        "played_mv",
        "kdl_policy",
        "final_v",
        "final_wdl_w",
        "final_wdl_d",
        "final_wdl_l",
        "final_moves_left",
        "zero_v",
        "zero_wdl_w",
        "zero_wdl_d",
        "zero_wdl_l",
        "zero_moves_left",
        "net_v",
        "net_wdl_w",
        "net_wdl_d",
        "net_wdl_l",
        "net_moves_left",
    ];

    fn to_vec(&self) -> Vec<f32> {
        let mut result = vec![
            self.game_id as f32,
            self.pos_index as f32,
            self.game_length as f32,
            self.zero_visits as f32,
            self.is_full_search as u8 as f32,
            self.is_final_position as u8 as f32,
            self.is_terminal as u8 as f32,
            self.hit_move_limit as u8 as f32,
            self.available_mv_count as f32,
            self.played_mv as f32,
            self.kdl_policy as f32,
        ];

        result.extend_from_slice(&self.final_values.to_slice());
        result.extend_from_slice(&self.zero_values.to_slice());
        result.extend_from_slice(&self.net_values.to_slice());

        assert_eq!(result.len(), Self::NAMES.len());
        result
    }
}
