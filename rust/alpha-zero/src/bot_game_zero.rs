use itertools::Itertools;
use rand::{Rng, thread_rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::ParallelIterator;
use board_game::ai::Bot;
use board_game::board::{Board, Outcome, Player};
use board_game::wdl::WDL;

use crate::network::Network;
use crate::selfplay::MoveSelector;
use crate::zero::{Request, Response, RunResult, Tree, ZeroSettings, ZeroState};

pub type OpponentConstructor<B> = Box<dyn Fn() -> Box<dyn Bot<B>> + Sync>;

struct GameState<B: Board> {
    opponent: usize,
    zero_first: bool,

    zero: ZeroState<B>,
    board: B,
    move_count: u32,
}

pub fn run<B: Board>(
    opponents: &[OpponentConstructor<B>],
    start_board: &B,
    iterations: u64,
    settings: ZeroSettings,
    network: &mut impl Network<B>,
    move_selector: MoveSelector,
    games_per_side: usize,
) -> Vec<Vec<WDL<usize>>> {
    let mut rng = thread_rng();

    // games in this vector are always either finished or with zero expecting a response
    // requests contains only the requests for the games that are not done yet, so it may have a different length
    let mut games = vec![];
    let mut requests = vec![];

    //start all games
    for opponent in 0..opponents.len() {
        for &zero_first in &[true, false] {
            for _ in 0..games_per_side {
                let mut board = start_board.clone();
                let mut move_count = 0;

                if !zero_first {
                    let mut opponent = opponents[opponent]();
                    board.play(opponent.select_move(&board));
                    move_count += 1;
                }

                let mut zero = ZeroState::new(Tree::new(board.clone()), iterations, settings);

                let result = zero.run_until_result(None, &mut rng);
                let request = unwrap_match!(result, RunResult::Request(req) => req);
                requests.push(request);

                games.push(GameState { opponent, zero_first, zero, board, move_count })
            }
        }
    }

    let mut prev_min_move_count = 0;

    loop {
        //check if we're done
        if requests.is_empty() { break; };

        let game_count = games.iter().filter(|g| !g.board.is_done()).count();
        let (min_move_count, max_move_count) = games.iter()
            .filter(|g| !g.board.is_done())
            .map(|g| g.move_count + 1)
            .minmax().into_option().unwrap();

        if min_move_count > prev_min_move_count {
            println!("Moves: {}-{}, games: {}", min_move_count, max_move_count, game_count);
            prev_min_move_count = min_move_count;
        }

        //evaluate requests
        let responses = network.evaluate_batch_requests(&requests);

        //continue playing games, using the responses
        let running_games = games.iter_mut()
            .filter(|g| !g.board.is_done())
            .collect_vec();
        let iter = running_games.into_par_iter()
            .zip_eq(responses.into_par_iter())
            .panic_fuse();
        requests = iter
            .filter_map(|(game, response)| {
                let mut rng = thread_rng();
                let opponent = opponents[game.opponent]();
                let request = game.run_until_request(opponent, response, iterations, settings, move_selector, &mut rng);
                request
            }).collect();
    }

    //collect results
    let mut wdls = vec![];

    for opponent in 0..opponents.len() {
        let mut opponent_wdls = vec![];

        for &zero_first in &[true, false] {
            let zero_player = if zero_first { Player::A } else { Player::B };
            let matches = games.iter()
                .filter(|g| g.opponent == opponent && g.zero_first == zero_first);

            let wins = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::WonBy(zero_player)).count();
            let draws = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::Draw).count();
            let losses = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::WonBy(zero_player.other())).count();

            assert_eq!(games_per_side, wins + draws + losses);

            println!("Opponent {}, zero first: {}, WDL: {} {} {}", opponent, zero_first, wins, draws, losses);

            opponent_wdls.push(WDL { win: wins, draw: draws, loss: losses })
        }

        wdls.push(opponent_wdls);
    }

    wdls
}

impl<B: Board> GameState<B> {
    /// Continue running this game until it's done or zero generates a new request.
    fn run_until_request(
        &mut self,
        mut opponent: Box<dyn Bot<B>>,
        response: Response<B>,
        iterations: u64,
        settings: ZeroSettings,
        move_selector: MoveSelector,
        rng: &mut impl Rng,
    ) -> Option<Request<B>> {
        //advance zero
        let result = self.zero.run_until_result(Some(response), rng);

        match result {
            RunResult::Request(request) => {
                Some(request)
            }
            RunResult::Done => {
                // select a move
                let index = move_selector.select(self.move_count, self.zero.tree.policy(), rng);
                let mv = self.zero.tree[self.zero.tree[0].children.unwrap().get(index)].last_move.unwrap();

                // play the move
                self.board.play(mv);
                self.move_count += 1;
                if self.board.is_done() { return None; }

                // play opponent move
                self.board.play(opponent.select_move(&self.board));
                self.move_count += 1;
                if self.board.is_done() { return None; }

                //start next zero search
                let mut zero = ZeroState::new(Tree::new(self.board.clone()), iterations, settings);
                let result = zero.run_until_result(None, rng);
                let request = unwrap_match!(result, RunResult::Request(req) => req);
                self.zero = zero;

                Some(request)
            }
        }
    }
}