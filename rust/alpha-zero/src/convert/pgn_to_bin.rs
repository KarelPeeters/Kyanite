use std::time::Instant;

use board_game::board::{Board, BoardAvailableMoves, Outcome, Player};
use board_game::games::chess::{ChessBoard, Rules};
use chess::ChessMove;
use internal_iterator::InternalIterator;

use pgn_reader::{PgnGame, PgnReader, PgnResult};
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
    pub allowed_time_controls: Option<Vec<String>>,
}

pub fn append_pgn_to_bin<M: BoardMapper<ChessBoard>>(
    input_pgn: impl BufferedReader<()>,
    binary_output: &mut BinaryOutput<ChessBoard, M>,
    filter: &Filter,
    max_games: Option<u32>,
    print: bool,
) -> Result<(), pgn_reader::Error> {
    let mut logger = Logger::new(max_games, print);

    let mut input = PgnReader::new(input_pgn);
    while let Some(game) = input.next()? {
        let mut skip_reasons = filter.decide_game(&game);

        if !skip_reasons.any() {
            let mut positions = vec![];
            let mut board = ChessBoard::default_with_rules(Rules::unlimited());

            let outcome = game.for_each_move(|mv| {
                let mv = board.parse_move(mv).unwrap();
                positions.push(build_position(&board, mv));
                board.play(mv);
            });

            let outcome = match outcome {
                PgnResult::WinWhite => Outcome::WonBy(Player::A),
                PgnResult::WinBlack => Outcome::WonBy(Player::B),
                PgnResult::Draw => Outcome::Draw,
                PgnResult::Star => unreachable!("Got * outcome, this game should already have been filtered out"),
            };

            skip_reasons.no_positions = positions.is_empty();

            if !positions.is_empty() {
                binary_output.append(Simulation { outcome, positions })?;
            }
        }

        logger.update(skip_reasons);
        if max_games.map_or(false, |max_games| logger.accepted_games >= max_games) {
            break;
        }
    }

    Ok(())
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
        match time_control {
            Some(time_control) => {
                self.allowed_time_controls.as_ref().map_or(true, |allowed| {
                    allowed.iter().any(|cand| cand == time_control)
                })
            }
            None => {
                return self.allowed_time_controls.is_none();
            }
        }
    }

    fn decide_game(&self, game: &PgnGame) -> SkipReasons {
        //TODO all of this is growing into a (slow) disaster, ideally skipping games is really fast
        //  also the star thing is super annoying, also breaking because there are no moves is annoying
        //  rewriting the filtering system

        let while_elo = game.header("WhiteElo").map(|x| x.parse::<u32>().unwrap());
        let black_elo = game.header("BlackElo").map(|x| x.parse::<u32>().unwrap());

        let skip_elo = !self.accept_elo(while_elo) || !self.accept_elo(black_elo);
        let skip_time_control = !self.accept_time_control(game.header("TimeControl"));
        let skip_termination = !accept_termination(game.header("Termination"));
        let result = game.header("Result").unwrap().parse::<PgnResult>().unwrap();
        let skip_result = PgnResult::Star == result;

        SkipReasons {
            elo: skip_elo,
            time_control: skip_time_control,
            termination: skip_termination,
            result: skip_result,
            no_positions: false,
        }
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

#[derive(Debug, Default, Copy, Clone)]
struct SkipReasons {
    elo: bool,
    time_control: bool,
    termination: bool,
    no_positions: bool,
    result: bool,
}

impl SkipReasons {
    fn any(&self) -> bool {
        self.elo || self.time_control || self.termination || self.result || self.no_positions
    }
}

#[derive(Debug)]
struct Logger {
    print: bool,
    max_games: Option<u32>,
    prev_update: Instant,

    visited_games: u32,
    accepted_games: u32,

    skipped_elo: u32,
    skipped_time_control: u32,
    skipped_termination: u32,
    skipped_result: u32,
    skipped_no_positions: u32,
}

impl Logger {
    fn new(max_games: Option<u32>, print: bool) -> Self {
        Logger {
            print,
            max_games,
            prev_update: Instant::now(),
            visited_games: 0,
            accepted_games: 0,
            skipped_elo: 0,
            skipped_time_control: 0,
            skipped_termination: 0,
            skipped_result: 0,
            skipped_no_positions: 0,
        }
    }

    fn update(&mut self, skip_reasons: SkipReasons) {
        self.visited_games += 1;
        self.accepted_games += (!skip_reasons.any()) as u32;

        self.skipped_elo += skip_reasons.elo as u32;
        self.skipped_time_control += skip_reasons.time_control as u32;
        self.skipped_termination += skip_reasons.termination as u32;
        self.skipped_result += skip_reasons.result as u32;
        self.skipped_no_positions += skip_reasons.no_positions as u32;

        // log progress
        let now = Instant::now();
        if self.print && (now - self.prev_update).as_secs_f32() > 1.0 {
            println!(
                "Accepted {}/{:?}, visited {}, skipped {} (result {}, elo: {}, tc {}, term {}, pos {})",
                self.accepted_games, self.max_games, self.visited_games, self.visited_games - self.accepted_games,
                self.skipped_result, self.skipped_elo, self.skipped_time_control, self.skipped_termination, self.skipped_no_positions,
            );
            self.prev_update = now;
        }
    }
}

fn build_position(board: &ChessBoard, mv: ChessMove) -> Position<ChessBoard> {
    let policy: Vec<f32> = board.available_moves()
        .map(|cand| (cand == mv) as u8 as f32)
        .collect();

    Position {
        board: board.clone(),
        should_store: true,
        zero_visits: 0,
        net_evaluation: ZeroEvaluation { values: ZeroValues::nan(), policy: vec![f32::NAN; policy.len()] },
        zero_evaluation: ZeroEvaluation { values: ZeroValues::nan(), policy },
    }
}
