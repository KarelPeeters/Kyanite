use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::ChessBoard;
use board_game::games::sttt::STTTBoard;
use board_game::games::ttt::TTTBoard;
use crossbeam::channel;
use crossbeam::channel::Sender;
use crossbeam::thread::Scope;
use itertools::Itertools;

use cuda_nn_eval::Device;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::sttt::STTTStdMapper;
use kz_core::mapping::ttt::TTTStdMapper;
use kz_core::mapping::BoardMapper;
use kz_core::network::muzero::MuZeroFusedGraphs;

use crate::server::collector::collector_main;
use crate::server::commander::{commander_main, read_command};
use crate::server::executor::{executor_loop_expander, executor_loop_root};
use crate::server::generator_muzero::generator_muzero_main;
use crate::server::job_channel::job_pair;
use crate::server::protocol::{Command, Game, GeneratorUpdate, Settings, StartupSettings};
use crate::server::rebatcher::Rebatcher;

pub fn selfplay_server_main() {
    println!("Waiting for connection");
    let (stream, addr) = TcpListener::bind("127.0.0.1:63105").unwrap().accept().unwrap();
    println!("Accepted connection {:?} on {:?}", stream, addr);

    let writer = BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    let startup_settings = wait_for_startup_settings(&mut reader);
    println!("Received startup settings:\n{:#?}", startup_settings);

    let cpu_games = startup_settings.cpu_batch_size * startup_settings.cpu_threads_per_device;
    let gpu_games = startup_settings.gpu_batch_size * startup_settings.gpu_threads_per_device;
    assert!(
        cpu_games >= gpu_games,
        "Not enough CPU games {} to fill potential concurrent GPU games {}",
        cpu_games,
        gpu_games,
    );

    let output_folder = Path::new(&startup_settings.output_folder);
    assert!(
        output_folder.exists(),
        "Output folder does not exist, got '{}'",
        startup_settings.output_folder
    );
    assert!(
        output_folder.is_absolute(),
        "Output folder is not an absolute path, got '{}'",
        startup_settings.output_folder
    );

    let game =
        Game::parse(&startup_settings.game).unwrap_or_else(|| panic!("Unknown game '{}'", startup_settings.game));

    //TODO static dispatch this early means we're generating a lot of code 4 times
    //  is it actually that much? -> investigate with objdump or similar
    //  would it be relatively easy to this dispatch some more?
    match game {
        Game::TTT => selfplay_start(game, startup_settings, TTTBoard::default, TTTStdMapper, reader, writer),
        Game::STTT => selfplay_start(
            game,
            startup_settings,
            STTTBoard::default,
            STTTStdMapper,
            reader,
            writer,
        ),
        Game::Ataxx { size } => selfplay_start(
            game,
            startup_settings,
            || AtaxxBoard::diagonal(size),
            AtaxxStdMapper::new(size),
            reader,
            writer,
        ),
        Game::Chess => selfplay_start(
            game,
            startup_settings,
            ChessBoard::default,
            ChessStdMapper,
            reader,
            writer,
        ),
    }
}

fn wait_for_startup_settings(reader: &mut BufReader<&TcpStream>) -> StartupSettings {
    match read_command(reader) {
        Command::StartupSettings(startup) => startup,
        command => panic!(
            "Must receive startup settings before any other command, got {:?}",
            command
        ),
    }
}

fn selfplay_start<B: Board>(
    game: Game,
    startup: StartupSettings,
    start_pos: impl Fn() -> B + Sync,
    mapper: impl BoardMapper<B> + 'static,
    reader: BufReader<impl Read + Send>,
    writer: BufWriter<impl Write + Send>,
) {
    let devices = Device::all().collect_vec();
    assert!(!devices.is_empty(), "No cuda devices found");

    let total_cpu_threads = startup.cpu_threads_per_device * devices.len();

    let mut settings_senders = vec![];
    let mut graph_senders = vec![];
    let (update_sender, update_receiver) = channel::bounded(total_cpu_threads);

    let start_pos = &start_pos;

    crossbeam::scope(|s| {
        // spawn cpu search, gpu eval, rebatcher threads
        for (device_id, &device) in devices.iter().enumerate() {
            let (mut new_settings_senders, mut new_graph_senders) =
                spawn_device_threads(s, device, device_id, &startup, mapper, start_pos, update_sender.clone());
            settings_senders.append(&mut new_settings_senders);
            graph_senders.append(&mut new_graph_senders);
        }

        // spawn collector
        s.builder()
            .name("collector".to_string())
            .spawn(move |_| {
                collector_main(
                    &game.to_string(),
                    writer,
                    startup.games_per_gen,
                    startup.first_gen,
                    &startup.output_folder,
                    mapper,
                    update_receiver,
                    total_cpu_threads,
                )
            })
            .unwrap();

        // spawn commander
        s.builder()
            .name("commander".to_string())
            .spawn(move |_| {
                commander_main(reader, mapper, settings_senders, graph_senders, update_sender);
            })
            .unwrap();

        // implicitly join all spawned threads
    })
    .unwrap();
}

fn spawn_device_threads<'s, B: Board, M: BoardMapper<B> + 'static, F: Fn() -> B + Sync>(
    s: &Scope<'s>,
    device: Device,
    device_id: usize,
    startup: &StartupSettings,
    mapper: M,
    start_pos: &'s F,
    update_sender: Sender<GeneratorUpdate<B>>,
) -> (Vec<Sender<Settings>>, Vec<Sender<Arc<MuZeroFusedGraphs<B, M>>>>) {
    let mut settings_senders: Vec<Sender<Settings>> = vec![];
    let mut graph_senders: Vec<Sender<Arc<MuZeroFusedGraphs<B, M>>>> = vec![];

    // TODO consider adding (non-blocking/best effort) rebatching to root executor as well
    let (root_client, root_server) = job_pair(2);
    let (rebatcher, expand_client, expand_server) = Rebatcher::new(2, startup.cpu_batch_size, startup.gpu_batch_size);

    let gpu_batch_size_expand = startup.gpu_batch_size;
    let gpu_batch_size_root = startup.gpu_batch_size_root;

    // spawn cpu threads
    for local_id in 0..startup.cpu_threads_per_device {
        let thread_id = startup.cpu_threads_per_device * device_id + local_id;

        let root_client = root_client.clone();
        let expand_client = expand_client.clone();
        let update_sender = update_sender.clone();

        let (settings_sender, settings_receiver) = channel::bounded(1);
        settings_senders.push(settings_sender);

        let cpu_batch_size = startup.cpu_batch_size;

        s.builder()
            .name(format!("generator-{}-{}", device_id, local_id))
            .spawn(move |_| {
                generator_muzero_main(
                    thread_id,
                    mapper,
                    start_pos,
                    cpu_batch_size,
                    settings_receiver,
                    root_client,
                    expand_client,
                    update_sender,
                )
            })
            .unwrap();
    }

    // spawn gpu expand eval threads
    for local_id in 0..startup.gpu_threads_per_device {
        let (graph_sender, graph_receiver) = channel::bounded(1);
        graph_senders.push(graph_sender);

        let expand_server = expand_server.clone();

        s.builder()
            .name(format!("gpu-expand-{}-{}", device_id, local_id))
            .spawn(move |_| {
                executor_loop_expander(device, gpu_batch_size_expand, graph_receiver, expand_server);
            })
            .unwrap();
    }

    // spawn gpu root eval thread
    {
        let (graph_sender, graph_receiver) = channel::bounded(1);
        graph_senders.push(graph_sender);

        let root_server = root_server.clone();

        s.builder()
            .name(format!("gpu-root-{}", device_id))
            .spawn(move |_| {
                executor_loop_root(device, gpu_batch_size_root, graph_receiver, root_server);
            })
            .unwrap();
    }

    // spawn rebatcher thread
    s.builder()
        .name(format!("rebatcher-{}", device_id))
        .spawn(move |_| {
            rebatcher.run_loop();
        })
        .unwrap();

    (settings_senders, graph_senders)
}
