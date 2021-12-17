use std::borrow::Cow;
use std::time::Instant;

use board_game::board::{Board, BoardAvailableMoves, Outcome, Player};
use board_game::games::chess::{ChessBoard, Rules};
use board_game::util::internal_ext::{Control, InternalIteratorExt};
use board_game::wdl::WDL;
use chess::ChessMove;
use internal_iterator::InternalIterator;

use pgn_reader::{PgnEval, PgnGame, PgnReader, PgnResult};
use pgn_reader::buffered_reader::BufferedReader;

use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;
use crate::network::ZeroEvaluation;
use crate::selfplay::simulation::{Position, Simulation};
use crate::zero::node::ZeroValues;

#[derive(Debug)]
pub struct Filter {
    pub min_elo: Option<u32>,
    pub max_elo: Option<u32>,
    pub min_start_time: Option<u32>,
    pub require_eval: bool,
}

pub fn append_pgn_to_bin<M: BoardMapper<ChessBoard>>(
    input_pgn: impl BufferedReader<()>,
    binary_output: &mut BinaryOutput<ChessBoard, M>,
    filter: &Filter,
    max_games: Option<u64>,
    print: bool,
) -> Result<(), pgn_reader::Error> {
    let mut logger = Logger::new(max_games, print);

    let mut time_input = 0.0;
    let mut time_moves = 0.0;
    let mut time_output = 0.0;

    let mut prev = Instant::now();

    let mut input = PgnReader::new(input_pgn);
    while let Some(game) = input.next()? {
        let mut skip = filter.should_skip(&game);

        time_input += time_since(&mut prev);
        let mut position_count = 0;

        if !skip {
            let mut positions = vec![];
            let mut board = ChessBoard::default_with_rules(Rules::unlimited());

            let outcome = match game.result() {
                PgnResult::WinWhite => Outcome::WonBy(Player::A),
                PgnResult::WinBlack => Outcome::WonBy(Player::B),
                PgnResult::Draw => Outcome::Draw,
                PgnResult::Star => unreachable!("Got * outcome, this game should already have been filtered out"),
            };

            let missing_eval = game.move_iter().for_each_control(|mv| {
                let eval = mv.field("eval").map(PgnEval::parse);
                if filter.require_eval && eval.is_none() {
                    Control::Break(())
                } else {
                    let mv = board.parse_move(mv.mv).unwrap();
                    positions.push(build_position(&board, mv, eval));
                    board.play(mv);

                    Control::Continue
                }
            }).is_some();

            time_moves += time_since(&mut prev);

            if missing_eval || positions.is_empty() {
                skip = true;
            } else {
                position_count = positions.len();
                binary_output.append(Simulation { outcome, positions })?;
            }

            time_output += time_since(&mut prev);
        }

        logger.update(skip, position_count, time_input, time_moves, time_output);
        if max_games.map_or(false, |max_games| logger.accepted_games >= max_games) {
            break;
        }
    }

    Ok(())
}

fn time_since(prev: &mut Instant) -> f32 {
    let now = Instant::now();
    let delta = (now - *prev).as_secs_f32();
    *prev = now;
    delta
}

impl Filter {
    pub fn accept_elo(&self, elo: Option<u32>) -> bool {
        match elo {
            Some(elo) => {
                self.min_elo.map_or(true, |min| min <= elo) &&
                    self.max_elo.map_or(true, |max| elo < max)
            }
            None => {
                self.min_elo.is_none() && self.max_elo.is_none()
            }
        }
    }

    pub fn accept_time_control(&self, time_control: Option<&str>) -> bool {
        match (self.min_start_time, time_control) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(min_start_time), Some(time_control)) => {
                time_control.find('+').map_or(false, |i| {
                    time_control[..i].parse::<u32>().unwrap() >= min_start_time
                })
            }
        }
    }

    fn should_skip(&self, game: &PgnGame) -> bool {
        // checks are ordered from likely to unlike skip reasons, and with short-circuiting

        // time control
        if self.min_start_time.is_some() && !self.accept_time_control(game.header("TimeControl")) {
            return true;
        }

        // elo
        if self.min_elo.is_some() || self.max_elo.is_some() {
            let while_elo = game.header("WhiteElo").map(|x| x.parse::<u32>().unwrap());

            if !self.accept_elo(while_elo) { return true; }
        }

        // termination & result
        !accept_termination(game.header("Termination")) ||
            game.header("Result").unwrap().parse::<PgnResult>().unwrap() == PgnResult::Star
    }
}

fn accept_termination(termination: Option<&str>) -> bool {
    match termination {
        Some("Normal") => true,
        Some("Time forfeit" | "Abandoned" | "Rules infraction" | "Unterminated") => false,
        _ => {
            eprintln!("Skipping unknown termination '{:?}'", termination);
            false
        }
    }
}

#[derive(Debug)]
struct Logger {
    print: bool,
    max_games: Option<u64>,
    prev_update: Instant,

    accepted_games: u64,
    skipped_games: u64,
    position_count: u64,
}

impl Logger {
    fn new(max_games: Option<u64>, print: bool) -> Self {
        Logger {
            print,
            max_games,
            prev_update: Instant::now(),

            accepted_games: 0,
            skipped_games: 0,
            position_count: 0,
        }
    }

    fn update(&mut self, skip: bool, position_count: usize, time_input: f32, time_moves: f32, time_output: f32) {
        self.accepted_games += (!skip) as u64;
        self.skipped_games += skip as u64;
        self.position_count += position_count as u64;

        // log progress
        let now = Instant::now();
        if self.print && (now - self.prev_update).as_secs_f32() > 1.0 {
            let total_games = self.accepted_games + self.skipped_games;
            println!(
                "Visited {}, converted {}/{:?}, positions: {}, skipped {} = {:.02}, time: (in {:.2} mv {:.2} out {:.2})",
                total_games, self.accepted_games, self.max_games, self.position_count,
                self.skipped_games, self.skipped_games as f32 / total_games as f32,
                time_input, time_moves, time_output
            );
            self.prev_update = now;
        }
    }
}

fn build_position(board: &ChessBoard, mv: ChessMove, eval: Option<PgnEval>) -> Position<ChessBoard> {
    let policy: Vec<f32> = board.available_moves()
        .map(|cand| (cand == mv) as u8 as f32)
        .collect();

    let zero_values = eval.map_or(ZeroValues::nan(), |eval| {
        let win = match board.next_player() {
            Player::A => eval.as_white_win_prob(),
            Player::B => 1.0 - eval.as_white_win_prob(),
        };

        ZeroValues {
            value: win * 2.0 - 1.0,
            wdl: WDL::new(win, 0.0, 1.0 - win),
        }
    });

    Position {
        board: board.clone(),
        should_store: true,
        zero_visits: 0,
        net_evaluation: ZeroEvaluation { values: ZeroValues::nan(), policy: Cow::Owned(vec![f32::NAN; policy.len()]) },
        zero_evaluation: ZeroEvaluation { values: zero_values, policy: Cow::Owned(policy) },
    }
}
