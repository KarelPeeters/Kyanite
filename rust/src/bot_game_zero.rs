use itertools::Itertools;
use rand::thread_rng;
use sttt::ai::Bot;
use sttt::board::{Board, Outcome, Player};

use crate::network::Network;
use crate::selfplay::MoveSelector;
use crate::zero::{RunResult, Tree, ZeroSettings, ZeroState};

struct GameState<B: Board> {
    opponent: usize,
    zero_first: bool,

    zero: ZeroState<B>,
    board: B,
    move_count: u32,
}

pub fn run<B: Board>(
    mut opponents: Vec<Box<dyn Bot<B>>>,
    start_board: &B,
    iterations: u64,
    settings: ZeroSettings,
    network: &mut impl Network<B>,
    move_selector: MoveSelector,
    games_per_side: usize,
) {
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
                    board.play(opponents[opponent].select_move(&board));
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
        let mut responses = network.evaluate_batch_requests(&requests);
        requests.clear();
        let mut response_iter = responses.drain(..);

        for game in &mut games {
            if game.board.is_done() { continue; }

            //advance zero
            let response = response_iter.next().unwrap();
            let result = game.zero.run_until_result(Some(response), &mut rng);

            match result {
                RunResult::Request(request) => {
                    requests.push(request)
                }
                RunResult::Done => {
                    // select a move
                    let index = move_selector.select(game.move_count, game.zero.tree.policy(), &mut rng);
                    let mv = game.zero.tree[game.zero.tree[0].children.unwrap().get(index)].last_move.unwrap();

                    // play the move
                    game.board.play(mv);
                    game.move_count += 1;
                    if game.board.is_done() { continue; }

                    // play opponent move
                    game.board.play(opponents[game.opponent].select_move(&game.board));
                    game.move_count += 1;
                    if game.board.is_done() { continue; }

                    //start next zero search
                    let mut zero = ZeroState::new(Tree::new(game.board.clone()), iterations, settings);
                    let result = zero.run_until_result(None, &mut rng);
                    requests.push(unwrap_match!(result, RunResult::Request(req) => req));
                    game.zero = zero;
                }
            }
        }
    }

    //collect results
    for opponent in 0..opponents.len() {
        for &zero_first in &[true, false] {
            let zero_player = if zero_first { Player::A } else { Player::B };
            let matches = games.iter()
                .filter(|g| g.opponent == opponent && g.zero_first == zero_first);

            let wins = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::WonBy(zero_player)).count();
            let draws = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::Draw).count();
            let losses = matches.clone().filter(|g| g.board.outcome().unwrap() == Outcome::WonBy(zero_player.other())).count();

            assert_eq!(games_per_side, wins + draws + losses);

            println!("Opponent {}, zero first: {}, WDL: {} {} {}", opponent, zero_first, wins, draws, losses);
        }
    }
}
