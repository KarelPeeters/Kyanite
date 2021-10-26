use std::borrow::Cow;
use std::io;
use std::time::Instant;

use board_game::board::{Board, BoardAvailableMoves, Outcome, Player};
use board_game::games::chess::{ChessBoard, moves_to_pgn, Rules};
use board_game::wdl::WDL;
use chess::{ChessMove, File, Piece, Rank, Square};
use internal_iterator::InternalIterator;
use pgn_lexer::parser::Token;
use shakmaty::{Chess, Move, Position as OtherPosition, Role};

use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;
use crate::network::ZeroEvaluation;
use crate::selfplay::simulation::{Position, Simulation};

#[derive(Debug)]
pub struct Filter {
    pub min_elo: Option<u32>,
    pub max_elo: Option<u32>,
    pub allowed_time_controls: Option<Vec<String>>,
}

impl Filter {
    pub fn accept_elo(&self, elo: u32) -> bool {
        self.min_elo.map_or(true, |min| min <= elo) &&
            self.max_elo.map_or(true, |max| elo < max)
    }

    pub fn accept_time_control(&self, time_control: &str) -> bool {
        self.allowed_time_controls.as_ref().map_or(true, |allowed| {
            allowed.iter().any(|a| a == time_control)
        })
    }
}

const IGNORED_SYMBOLS: &[&str] = &[
    "Event", "Site", "Date", "Round", "White", "Black", "Result", "UTCDate", "UTCTime",
    "WhiteRatingDiff", "BlackRatingDiff", "ECO", "Opening", "BlackTitle", "WhiteTitle"
];

pub fn append_pgn_to_bin<M: BoardMapper<ChessBoard>>(
    input_pgn: &[u8],
    binary_output: &mut BinaryOutput<ChessBoard, M>,
    filter: &Filter,
    max_games: Option<u32>,
    log: bool,
) -> io::Result<()> {
    let mut prev_update = Instant::now();
    let mut game_count = 0;
    let mut skipped = 0;
    let mut wrong_elo = 0;
    let mut wrong_tc = 0;
    let mut wrong_termination = 0;
    let mut read_index = 0;

    let mut keep_game = false;
    let mut positions = vec![];
    let mut curr_board = ChessBoard::default_with_rules(Rules::unlimited());

    let mut tag_symbol = None;

    for token in pgn_lexer::parser::PGNTokenIterator::new(input_pgn) {
        let now = Instant::now();
        if log && (now - prev_update).as_secs_f32() > 1.0 {
            println!(
                "game count: {}/{:?}, skipped {} (elo: {}, tc {}, term {}), file read: {:.4}",
                game_count, max_games, skipped, wrong_elo, wrong_tc, wrong_termination,
                read_index as f32 / input_pgn.len() as f32,
            );
            prev_update = now;
        }

        if !keep_game {
            if let Token::Result(_) = token {
                keep_game = true;
            }
            continue;
        }

        match token {
            Token::TagString(value) => {
                read_index = subslice_start(input_pgn, value).unwrap();

                let symbol = tag_symbol.take().expect("Got tag string without symbol");
                let value = std::str::from_utf8(value).unwrap();

                let mut is_elo = false;
                let mut is_tc = false;
                let mut is_termination = false;

                let delta_keep_game = match symbol {
                    "WhiteElo" | "BlackElo" => {
                        is_elo = true;
                        filter.accept_elo(value.parse().unwrap())
                    }
                    "TimeControl" => {
                        is_tc = true;
                        filter.accept_time_control(value)
                    }
                    "Termination" => {
                        is_termination = true;
                        match value {
                            "Normal" => true,
                            "Time forfeit" | "Abandoned" | "Rules infraction" | "Unterminated" => false,
                            _ => {
                                eprintln!("Skipping unknown termination '{}'", value);
                                false
                            }
                        }
                    }
                    _ => {
                        if !IGNORED_SYMBOLS.contains(&symbol) {
                            eprintln!("Ignoring unknown tag '{}: {}'", symbol, value);
                        }
                        true
                    }
                };

                if keep_game & !delta_keep_game {
                    skipped += 1;

                    wrong_elo += is_elo as u32;
                    wrong_tc += is_tc as u32;
                    wrong_termination += is_termination as u32;
                }
                keep_game &= delta_keep_game;
            }

            Token::Move(mv) => {
                let mv = std::str::from_utf8(mv).unwrap();

                let mv = parse_mv(curr_board.inner(), mv).unwrap_or_else(|e| {
                    eprintln!("Failed to parse move '{}' with error {:?} on board\n{}", mv, e, curr_board);
                    panic!();
                });

                // with the move we now know the policy of the previous position, and we can store it
                positions.push(build_position(&curr_board, mv));
                curr_board.play(mv);
            }

            Token::Result(result) => {
                if positions.is_empty() {
                    continue;
                }

                let result = std::str::from_utf8(result).unwrap();
                let outcome = match result {
                    "1/2-1/2" => Some(Outcome::Draw),
                    "1-0" => Some(Outcome::WonBy(Player::A)),
                    "0-1" => Some(Outcome::WonBy(Player::B)),
                    _ => {
                        eprintln!("Skipping unknown result '{}'", result);
                        None
                    }
                };

                if let Some(outcome) = outcome {
                    game_count += 1;

                    let simulation = Simulation { outcome, positions };
                    positions = vec![];
                    binary_output.append(simulation)?;
                }

                positions.clear();
                curr_board = ChessBoard::default_with_rules(Rules::unlimited());
            }

            // bookkeeping states
            Token::NullMove(_) => panic!("null move"),
            Token::EscapeComment(_) | Token::NAG(_) | Token::MoveAnnotation(_) | Token::Commentary(_) => {}
            Token::TagSymbol(symbol) => {
                assert!(tag_symbol.is_none());
                tag_symbol = Some(std::str::from_utf8(symbol).unwrap());
            }
            Token::StartVariation(_) => panic!("variation"),
            Token::EndVariation(_) => panic!("variation"),
        }

        if max_games.map_or(false, |max_games| game_count >= max_games) {
            break;
        }
    }

    Ok(())
}

fn parse_mv(board: &chess::Board, mv: &str) -> Result<ChessMove, chess::Error> {
    // the chess crate move parsing is kind of strange we need to help it a bit
    let removed_chars: &[char] = &['=', '+', '#'];
    let mv = if mv.contains(removed_chars) {
        Cow::from(mv.replace(removed_chars, ""))
    } else {
        Cow::from(mv)
    };

    match ChessMove::from_san(&board, &mv) {
        Ok(mv) => Ok(mv),
        Err(original_err) => {
            // try appending e.p. to get it to parse an en passant move
            let mv_ep = mv.into_owned() + " e.p.";
            ChessMove::from_san(&board, &mv_ep).map_err(|_| original_err)
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
        zero_evaluation: ZeroEvaluation { wdl: WDL::nan(), policy: vec![f32::NAN; policy.len()] },
        net_evaluation: ZeroEvaluation { wdl: WDL::nan(), policy },
    }
}

fn subslice_start<T>(slice: &[T], sub_slice: &[T]) -> Option<usize> {
    if slice.as_ptr_range().contains(&sub_slice.as_ptr()) {
        // safety: we just asserted that the subslice starts within the main slice
        unsafe {
            Some(sub_slice.as_ptr().offset_from(slice.as_ptr()) as usize)
        }
    } else {
        None
    }
}

struct ToBinVisitor<'a, M: BoardMapper<ChessBoard>> {
    log: bool,
    rules: Rules,
    binary_output: &'a mut BinaryOutput<ChessBoard, M>,

    min_elo: Option<u32>,
    max_elo: Option<u32>,
    max_games: Option<u32>,

    elo: Option<u32>,
    skip_next: bool,

    curr_board: ChessBoard,
    curr_board_other: Chess,
    positions: Option<Vec<Position<ChessBoard>>>,
    moves: Vec<ChessMove>,
    skipped_games: u32,
}
