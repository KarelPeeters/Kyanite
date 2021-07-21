use rand::thread_rng;

use sttt::ai::simple::RandomBot;
use sttt::uai;

fn main() -> std::io::Result<()> {
    uai::client::run(RandomBot::new(thread_rng()))
}