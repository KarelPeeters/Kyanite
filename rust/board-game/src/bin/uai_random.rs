use rand::thread_rng;

use board_game::ai::simple::RandomBot;
use board_game::uai;

fn main() -> std::io::Result<()> {
    uai::client::run(RandomBot::new(thread_rng()))
}