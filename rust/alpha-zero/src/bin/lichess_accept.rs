use std::time::Duration;

use tokio_stream::StreamExt;

use licoricedev::client::{Lichess, LichessResult};
use licoricedev::models::board::Event;

fn main() -> LichessResult<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { main_impl().await })
}

async fn main_impl() -> LichessResult<()> {
    let token = std::fs::read_to_string("ignored/lichess_token.txt")?;
    let lichess = Lichess::new(token);

    // infinite loop because sometimes the stream randomly closes (probably lichess maintenance?)
    loop {
        let mut stream = lichess.stream_incoming_events().await?;
        while let Some(event) = stream.next().await {
            let event = event?;
            if let Event::Challenge { challenge } = event {
                println!("Got challenge");
                println!("  from:    {:?}", challenge.challenger);
                println!("  tc:      {:?}", challenge.time_control);
                println!("  variant: {:?}", challenge.variant);
                println!("  fen:     {:?}", challenge.initial_fen);

                if challenge.variant.key != "standard" {
                    println!("Declined");
                    if let Err(e) = lichess.challenge_decline(&challenge.id, Some("This bot does not play variants")).await {
                        println!("Error: {:?}", e)
                    }

                    continue;
                }

                println!("Accepted");
                if let Err(e) = lichess.challenge_accept(&challenge.id).await {
                    println!("Error: {:?}", e)
                }
            }
        }

        std::thread::sleep(Duration::from_secs(10));
    }
}
