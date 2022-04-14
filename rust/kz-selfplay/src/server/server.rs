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

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::sttt::STTTStdMapper;
use kz_core::mapping::ttt::TTTStdMapper;
use kz_core::mapping::BoardMapper;

use crate::server::collector::collector_main;
use crate::server::commander::{commander_main, read_command};
use crate::server::protocol::{Command, Game, GeneratorUpdate, Settings, StartupSettings};
use crate::server::server_alphazero::AlphaZeroSpecialization;
use crate::server::server_muzero::MuZeroSpecialization;

pub fn selfplay_server_main() {
    println!("Waiting for connection");
    let (stream, addr) = TcpListener::bind("127.0.0.1:63105").unwrap().accept().unwrap();
    println!("Accepted connection {:?} on {:?}", stream, addr);

    let writer = BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    let startup_settings = wait_for_startup_settings(&mut reader);
    println!("Received startup settings:\n{:#?}", startup_settings);

    assert_ne!(startup_settings.cpu_batch_size, 0, "CPU batch size must be nonzero");
    assert_ne!(startup_settings.gpu_batch_size, 0, "GPU batch size must be nonzero");
    if startup_settings.muzero {
        assert_ne!(
            startup_settings.gpu_batch_size_root, 0,
            "For muzero root batch size must be nonzero"
        );
    }

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

fn selfplay_start<B: Board, M: BoardMapper<B> + 'static, F: Fn() -> B + Sync>(
    game: Game,
    startup: StartupSettings,
    start_pos: F,
    mapper: M,
    reader: BufReader<impl Read + Send>,
    writer: BufWriter<impl Write + Send>,
) {
    if startup.muzero {
        selfplay_start_impl(game, startup, mapper, start_pos, reader, writer, MuZeroSpecialization);
    } else {
        selfplay_start_impl(
            game,
            startup,
            mapper,
            start_pos,
            reader,
            writer,
            AlphaZeroSpecialization,
        );
    }
}

pub trait ZeroSpecialization<B: Board, M: BoardMapper<B> + 'static> {
    type G: Send + Sync;

    fn spawn_device_threads<'s>(
        &self,
        s: &Scope<'s>,
        device: Device,
        device_id: usize,
        startup: &StartupSettings,
        mapper: M,
        start_pos: &'s (impl Fn() -> B + Sync),
        update_sender: Sender<GeneratorUpdate<B>>,
    ) -> (Vec<Sender<Settings>>, Vec<Sender<Arc<Self::G>>>);

    fn load_graph(&self, path: &str, mapper: M) -> Self::G;
}

fn selfplay_start_impl<B: Board, M: BoardMapper<B> + 'static, Z: ZeroSpecialization<B, M> + Send + Sync>(
    game: Game,
    startup: StartupSettings,
    mapper: M,
    start_pos: impl Fn() -> B + Sync,
    reader: BufReader<impl Read + Send>,
    writer: BufWriter<impl Write + Send>,
    spec: Z,
) {
    let devices = Device::all().collect_vec();
    assert!(!devices.is_empty(), "No cuda devices found");

    let total_cpu_threads = startup.cpu_threads_per_device * devices.len();

    let mut settings_senders = vec![];
    let mut graph_senders: Vec<Sender<Arc<Z::G>>> = vec![];
    let (update_sender, update_receiver) = channel::bounded(total_cpu_threads);

    let start_pos = &start_pos;

    crossbeam::scope(|s| {
        // spawn cpu search, gpu eval, rebatcher threads
        for (device_id, &device) in devices.iter().enumerate() {
            let (mut new_settings_senders, mut new_graph_senders) =
                spec.spawn_device_threads(s, device, device_id, &startup, mapper, start_pos, update_sender.clone());
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
                commander_main(reader, settings_senders, graph_senders, update_sender, |path| {
                    spec.load_graph(path, mapper)
                });
            })
            .unwrap();

        // implicitly join all spawned threads
    })
    .unwrap();
}
