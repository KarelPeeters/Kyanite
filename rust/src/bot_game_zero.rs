use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::{Rng, thread_rng};
use rand_distr::WeightedIndex;
use sttt::board::{Board, Coord, Player};
use sttt::bot_game::Bot;

use crate::network::Network;
use crate::zero::{RunResult, Tree, ZeroSettings, ZeroState};

struct GameState {
    opponent: usize,
    zero_first: bool,

    zero: ZeroState,
    board: Board,
}

pub fn run(
    mut opponents: Vec<Box<dyn Bot>>,
    iterations: u64,
    settings: ZeroSettings,
    network: &mut impl Network,
    temperature: f32,
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
                let mut board = Board::new();

                if !zero_first {
                    board.play(opponents[opponent].play(&board).unwrap());
                }

                let mut zero = ZeroState::new(Tree::new(board.clone()), iterations, settings);

                let result = zero.run_until_result(None, &mut rng);
                let request = unwrap_match!(result, RunResult::Request(req) => req);
                requests.push(request);

                games.push(GameState { opponent, zero_first, zero, board })
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
            .map(|g| g.board.count_tiles() + 1)
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
                    // play zero's move
                    game.board.play(pick_move(&game.zero.tree, temperature, &mut rng));
                    if game.board.is_done() { continue; }

                    // play opponent move
                    game.board.play(opponents[game.opponent].play(&game.board).unwrap());
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
            let zero_player = if zero_first { Player::X } else { Player::O };
            let matches = games.iter()
                .filter(|g| g.opponent == opponent && g.zero_first == zero_first);

            let wins = matches.clone().filter(|g| g.board.won_by.unwrap() == zero_player).count();
            let draws = matches.clone().filter(|g| g.board.won_by.unwrap() == Player::Neutral).count();
            let losses = matches.clone().filter(|g| g.board.won_by.unwrap() == zero_player.other()).count();

            assert_eq!(games_per_side, wins + draws + losses);

            println!("Opponent {}, zero first: {}, WDL: {} {} {}", opponent, zero_first, wins, draws, losses);
        }
    }
}

fn pick_move(tree: &Tree, temperature: f32, rng: &mut impl Rng) -> Coord {
    let index = if temperature == 0.0 {
        tree.policy().map(OrderedFloat).position_max().unwrap()
    } else {
        let policy_temp = tree.policy().map(|p| p.powf(1.0 / temperature));
        let distr = WeightedIndex::new(policy_temp).unwrap();
        rng.sample(distr)
    };
    tree[tree[0].children.unwrap().get(index)].coord.unwrap()
}