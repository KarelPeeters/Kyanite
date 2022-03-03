#![allow(unused_imports)]
#![allow(unused_macros)]

use chrono::{TimeZone, Utc};
use futures_util::TryStreamExt;
use licorice::client::{Lichess, LichessResult};
use licorice::models::game::Player;
use licorice::models::user::PerfType;
use serde_json::to_string_pretty;
use std::{env, thread, time};

#[tokio::main]
async fn main() -> LichessResult<()> {
    // let lichess = Lichess::default();
    let lichess = Lichess::new(env::var("LICHESS_PAT_0").unwrap());

    macro_rules! rstr {
        ($lifunc:ident($($param:expr),*)) => {
            println!("{}", lichess.$lifunc($($param,)*).await?);
        };
    }

    macro_rules! rbyts {
        ($lifunc:ident($($param:expr),*)) => {
            let mut stream = lichess.$lifunc($($param,)*).await?;
            while let Some(byte) = stream.try_next().await? {
                print!("{:?}", byte);
            }
        };
    }

    macro_rules! fmod {
        ($lifunc:ident($($param:expr),*)) => {
            println!(
                "{}",
                to_string_pretty(&lichess.$lifunc($($param,)*).await?)?
            );
        };
    }

    macro_rules! smod {
        ($lifunc:ident($($param:expr),*)) => {
            let mut stream = lichess.$lifunc($($param,)*).await?;
            while let Some(model) = stream.try_next().await? {
                println!("{}", to_string_pretty(&model)?);
            }
        };
    }

    // // Account

    // fmod!(get_my_profile());
    let myself = lichess.get_my_profile().await?;
    // println!("{}", lichess.get_my_email_address().await?);
    // fmod!(get_my_preferences());
    // println!("{}", lichess.get_my_kid_mode_status().await?);
    // lichess.set_my_kid_mode_status(false).await?;

    // // Users

    // fmod!(get_user_status(&["minhjtran", "mutdpro", "mircica"]));
    // fmod!(get_all_top_10());
    // fmod!(get_one_leaderboard(5, PerfType::KingOfTheHill));
    // fmod!(get_user_public("zhigalko_sergei"));
    // fmod!(get_rating_history("icp1994"));
    // smod!(get_my_puzzle_activity(Some(100)));
    // fmod!(get_users_by_ids(&["minhjtran", "mutdpro", "mircica"]));
    // smod!(get_members_of_a_team("melkumyan-fan-club"));
    // fmod!(get_live_streamers());
    // fmod!(get_crosstable("namiro", "vostanin", true));

    // // Relations

    // smod!(get_followings("mircica"));
    // smod!(get_followers("mrhaggis"));

    // // Games

    let since = Utc
        .ymd(2018, 12, 25)
        .and_hms(0, 0, 0)
        .timestamp_millis()
        .to_string();
    let until = Utc
        .ymd(2019, 2, 17)
        .and_hms(0, 0, 0)
        .timestamp_millis()
        .to_string();
    let query_params = vec![
        ("evals", "false"),
        ("moves", "false"),
        ("max", "100"),
        ("since", &since),
        ("until", &until),
    ];
    // rstr!(export_one_game_pgn("4kU9hKhl", Some(&query_params)));
    // fmod!(export_one_game_json("4W3qyO9R", None));
    // fmod!(export_one_game_json("eO8hev6Y", None));
    // fmod!(export_one_game_json("kaMxZzMD", None));
    // rstr!(export_ongoing_game_pgn("holdenhc", None));
    // fmod!(export_ongoing_game_json("holdenhc", Some(&query_params)));
    // smod!(export_all_games_json("mita_m", Some(&query_params)));
    let mut stream = lichess
        .export_all_games_json(&myself.id, Some(&query_params))
        .await?;
    while let Some(game) = stream.try_next().await? {
        let players = game.players;
        if let Player::Entity(white_entity) = players.white {
            if white_entity.user.unwrap().username == myself.username {
                if let Player::Entity(black_entity) = players.black {
                    println!("Played White in {} vs {}", game.id, black_entity.rating)
                }
            } else {
                println!("Played Black in {} vs {}", game.id, white_entity.rating)
            }
        }
    }
    // rbyts!(export_all_games_pgn("mita_m", Some(&query_params)));
    // smod!(export_games_by_ids_json(
    //     &["4kU9hKhl", "4W3qyO9R", "eO8hev6Y", "kaMxZzMD"],
    //     Some(&query_params)
    // ));
    // smod!(stream_current_games(&["ruchess27", "rmnp",]));
    // smod!(stream_current_games(
    //     &"nikolai_69,rus5".split(',').collect::<Vec<&str>>()
    // ));
    // fmod!(get_ongoing_games(10));
    // fmod!(get_current_tv_games());
    // let pgn = std::fs::read_to_string("./examples/export_game.pgn")?;
    // println!("{}", lichess.import_one_game(&pgn).await?);

    // // Teams

    // smod!(get_team_swiss_tournaments("chessnetwork", 50));
    // fmod!(get_a_single_team("chessnetwork"));
    // fmod!(get_popular_teams(None));
    // fmod!(teams_of_a_player("chess-network"));
    // fmod!(search_teams("network", None));
    // smod!(get_team_arena_tournaments(
    //     "igm-gata-kamskys-pawngrabbers-club",
    //     15
    // ));
    // lichess
    //     .join_a_team("heartlecc", Some("pretty please!"))
    //     .await?;
    // lichess.leave_a_team("lichess-swiss").await?;
    // lichess
    //     .kick_user_from_team("chessnetwork", "chess-network")
    //     .await?;
    // lichess
    //     .message_all_members("chessnetwork", "hope this doesn't work!")
    //     .await?;

    // // Board

    // smod!(stream_incoming_events());
    // rbyts!(create_a_seek(120, 30, None));
    // let form_params = [("color", "white")];
    // rbyts!(create_a_seek(120, 30, Some(&form_params)));

    // // Challenges

    // let form_params = [
    //     ("clock.limit", "180"),
    //     ("clock.increment", "2"),
    //     // ("days", "2"),
    //     ("color", "black"),
    //     (
    //         "fen",
    //         "rnbqkb1r/1p3ppp/p1p1pn2/3p4/P1PP4/2N2N2/1P2PPPP/R1BQKB1R w KQkq - 0 6",
    //     ),
    //     // ("acceptByToken", &env::var("LICHESS_PAT_1")?),
    // ];
    // fmod!(challenge_create("icp1994", Some(&form_params)));
    // let challenge = lichess
    //     .challenge_create("icp1994", Some(&form_params))
    //     .await?;
    // let cid = challenge.challenge.unwrap().id;
    // println!("{}", &cid);
    // let gid = challenge.game.unwrap().id;
    // println!("{}", &gid);
    // let user = Lichess::new(std::env::var("LICHESS_PAT_1").unwrap());
    // thread::sleep(time::Duration::from_secs(30));
    // user.challenge_accept(&cid).await?;
    // user.challenge_decline(&cid).await?;
    // user.challenge_cancel(&cid).await?;
    // fmod!(challenge_stockfish(2, None));
    // fmod!(challenge_open(None));
    // lichess
    //     .start_game_clocks(
    //         &gid,
    //         &env::var("LICHESS_PAT_0").unwrap(),
    //         &env::var("LICHESS_PAT_1").unwrap(),
    //     )
    //     .await?;

    // // Arena

    // fmod!(arena_current());
    // fmod!(arena_new(3u16, 2u8, 60u16, None));
    // fmod!(arena_info("2yenmFSs", 5));
    // rbyts!(games_by_arena_pgn("h06zb5YN", None));
    // smod!(games_by_arena_json("h06zb5YN", None));
    // smod!(results_by_arena("A2ojpt9J", 20));
    // fmod!(teams_by_arena("0syvGIIT"));
    // smod!(arenas_by_user("nojoke", 20));

    // // Swiss

    // fmod!(swiss_new("heartlecc", 180, 2, 4, None));
    // rstr!(swiss_trf("3c3VDNcI"));
    // rbyts!(games_by_swiss_pgn("3c3VDNcI", None));
    // smod!(games_by_swiss_json("3c3VDNcI", None));
    // smod!(results_by_swiss("mNimKV8p", 30));

    // // Broadcast

    // smod!(get_official_broadcasts(20));
    // let broadcast = lichess
    //     .create_broadcast("Licorice Test Broadcast", "For testing purpose", None)
    //     .await?;
    // fmod!(get_broadcast(&broadcast.id));
    // fmod!(update_broadcast(
    //     &broadcast.id,
    //     "Name Changer",
    //     "New Description",
    //     None
    // ));
    // lichess
    //     .push_to_broadcast(&broadcast.id, include_str!("export_game.pgn"))
    //     .await?;

    // // Misc

    // fmod!(get_current_simuls());
    // rstr!(study_chapter_pgn("h0yrcKtz", "5MZg70J3", None));
    // rstr!(study_full_pgn("Of2YrR6A", None));

    Ok(())
}
