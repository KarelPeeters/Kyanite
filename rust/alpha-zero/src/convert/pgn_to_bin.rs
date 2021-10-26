use std::io;
use std::io::Read;

use board_game::board::{Board, BoardAvailableMoves, Outcome, Player};
use board_game::games::chess::{ChessBoard, moves_to_pgn, Rules};
use board_game::wdl::WDL;
use chess::{ChessMove, File, Piece, Rank, Square};
use internal_iterator::InternalIterator;
use pgn_reader::{BufferedReader, Color, RawHeader, SanPlus, Skip, Visitor};
use shakmaty::{Chess, Move, Position as OtherPosition, Role};

use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;
use crate::network::ZeroEvaluation;
use crate::selfplay::simulation::{Position, Simulation};

pub fn append_pgn_to_bin<M: BoardMapper<ChessBoard>>(
    rules: Rules,
    input_pgn: impl Read,
    binary_output: &mut BinaryOutput<ChessBoard, M>,
    min_elo: Option<u32>,
    max_elo: Option<u32>,
    max_games: Option<u32>,
    log: bool,
) -> io::Result<()> {
    let mut visitor = ToBinVisitor {
        log,
        rules,
        binary_output,
        min_elo,
        max_elo,
        max_games,
        elo: None,
        skip_next: true,
        curr_board: ChessBoard::default_with_rules(rules),
        curr_board_other: Chess::default(),
        positions: None,
        moves: vec![],
        skipped_games: 0,
    };

    let mut reader = BufferedReader::new(input_pgn);
    reader.read_all(&mut visitor)
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

impl<M: BoardMapper<ChessBoard>> Visitor for ToBinVisitor<'_, M> {
    type Result = ();

    fn begin_headers(&mut self) {
        self.skip_next = false;
        self.elo = None;
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let key = std::str::from_utf8(key)
            .expect("Failed to parse header key");
        let value = value.decode_utf8()
            .expect("Failed to parse header value");

        match key {
            "Site" => {
                // println!("{}", value);
            }
            "Variant" => {
                self.skip_next = true;
            }
            "WhiteElo" | "BlackElo" => {
                let elo = value.parse::<u32>()
                    .expect("Failed to parse elo value as int");

                // store average elo
                self.elo = Some(self.elo.map_or(elo, |other_elo| (elo + other_elo) / 2));
            }
            "Termination" => {
                match &*value {
                    "Time forfeit" | "Abandoned" | "Unterminated" | "" => { self.skip_next = true }
                    "Normal" => (),
                    other => {
                        self.skip_next = true;
                        println!("Skipping unknown termination '{}'", other);
                    }
                }
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        assert!(self.positions.is_none());

        let mut skip_elo = false;
        if let Some(min_elo) = self.min_elo {
            skip_elo |= self.elo.map_or(true, |elo| elo < min_elo);
        }
        if let Some(max_elo) = self.max_elo {
            skip_elo |= self.elo.map_or(true, |elo| elo >= max_elo);
        }

        let skip_game_count = self.max_games
            .map_or(false, |max_games| self.binary_output.game_count() >= max_games as usize);

        let skip = self.skip_next || skip_elo || skip_game_count;

        if skip {
            self.skipped_games += 1;
        } else {
            self.positions = Some(vec![]);
            self.curr_board = ChessBoard::default_with_rules(self.rules);
            self.curr_board_other = Chess::default();
        }
        Skip(skip)
    }

    fn san(&mut self, san_plus: SanPlus) {
        let mv_other = san_plus.san.to_move(&self.curr_board_other)
            .unwrap_or_else(|e| panic!("Failed to parse move {} for board {}: {:?}", san_plus.san, self.curr_board, e));

        let mv = map_mv(mv_other.clone());
        self.moves.push(mv);

        if self.curr_board.is_done() || !self.curr_board.is_available_move(mv) {
            eprintln!("{}", moves_to_pgn(&self.moves));
            eprintln!("{}", self.curr_board);

            assert!(
                !self.curr_board.is_done(),
                "Board is already done, tried to parse new move {}", mv
            );

            assert!(
                self.curr_board.is_available_move(mv),
                "Parsed unavailable move {} on board {}", mv, self.curr_board
            );
        }

        //one-hot encode the policy
        let policy: Vec<f32> = self.curr_board.available_moves()
            .map(|cand| (cand == mv) as u8 as f32)
            .collect();

        let position = Position {
            board: self.curr_board.clone(),
            should_store: true,

            zero_visits: 0,
            net_evaluation: ZeroEvaluation { wdl: WDL::nan(), policy: vec![f32::NAN; policy.len()] },
            zero_evaluation: ZeroEvaluation { wdl: WDL::nan(), policy },
        };

        //append the position from before the move, so we always have a policy
        self.positions.as_mut().unwrap().push(position);

        self.curr_board.play(mv);
        self.curr_board_other = self.curr_board_other.clone().play(&mv_other).unwrap();
    }

    fn outcome(&mut self, outcome: Option<shakmaty::Outcome>) {
        let positions = self.positions.take().unwrap();

        let pgn_outcome = match outcome {
            None => None,
            Some(shakmaty::Outcome::Draw) => Some(Outcome::Draw),
            Some(shakmaty::Outcome::Decisive { winner: Color::White }) => Some(Outcome::WonBy(Player::A)),
            Some(shakmaty::Outcome::Decisive { winner: Color::Black }) => Some(Outcome::WonBy(Player::B)),
        };

        let expected_outcome = self.curr_board.outcome();

        let (outcome, consistent) = match (pgn_outcome, expected_outcome) {
            (Some(p), Some(e)) => (Some(e), p == e),
            (None, Some(e)) => (Some(e), false),
            // this just means one of the bots resigned
            //  what about draws? investigate some more!
            (Some(p), None) => (Some(p), true),
            (None, None) => (None, false)
        };

        if !consistent {
            eprintln!("Inconsistent game outcome, pgn contains {:?}, expected {:?}", pgn_outcome, expected_outcome);
            eprintln!("Assuming outcome {:?}", outcome);
            eprintln!("Moves: {}", moves_to_pgn(&self.moves));
            eprintln!("Board: {}", self.curr_board);
        }

        if let Some(outcome) = outcome {
            if !positions.is_empty() {
                let simulation = Simulation { positions, outcome };
                self.binary_output.append(simulation)
                    .expect("Error during simulation appending");
            }
        }

        if self.log && self.binary_output.game_count() % 1000 == 0 {
            println!("games: saved {}/{:?}, skipped: {}", self.binary_output.game_count(), self.max_games, self.skipped_games)
        }

        self.moves.clear();
    }

    fn end_game(&mut self) -> Self::Result {
        ()
    }
}

fn map_square(sq: shakmaty::Square) -> Square {
    Square::make_square(
        Rank::from_index(sq.rank() as usize),
        File::from_index(sq.file() as usize),
    )
}

fn map_piece(role: Role) -> Piece {
    match role {
        Role::Pawn => Piece::Pawn,
        Role::Knight => Piece::Knight,
        Role::Bishop => Piece::Bishop,
        Role::Rook => Piece::Rook,
        Role::Queen => Piece::Queen,
        Role::King => Piece::King,
    }
}

fn map_mv(mv: Move) -> ChessMove {
    match mv {
        Move::Normal { role: _, from, capture: _, to, promotion } => {
            ChessMove::new(
                map_square(from),
                map_square(to),
                promotion.map(|role| map_piece(role)),
            )
        }
        Move::EnPassant { from, to } => {
            ChessMove::new(
                map_square(from),
                map_square(to),
                None,
            )
        }
        Move::Castle { king, rook } => {
            let from = map_square(king);

            let direction = (rook.file() as i8 - king.file() as i8).signum();
            let to_file = (from.get_file().to_index() as i8 + 2 * direction) as usize;
            let to = Square::make_square(from.get_rank(), File::from_index(to_file));

            ChessMove::new(from, to, None)
        }
        Move::Put { .. } => panic!("There are no Put moves in normal chess")
    }
}