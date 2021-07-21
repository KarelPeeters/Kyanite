#![allow(unused_variables)]
#![allow(dead_code)]

use std::cmp::min;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;

use crossbeam::channel;
use crossbeam::channel::Sender;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tch::Device;

use alpha_zero::network::torch_utils::all_cuda_devices;

#[derive(Debug, Serialize, Deserialize, Clone)]
enum Command {
    Stop,
    NewSettings { settings: Settings },
}

#[derive(Debug)]
struct StartupSettings {
    output_folder: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Settings {
    // network
    network_path: String,

    // performance
    games_per_file: u32,
    games_per_thread: u32,
    threads_per_device: u32,

    // selfplay
    temperature: f32,
    zero_temp_move_count: u32,

    dirichlet_alpha: f32,
    dirichlet_eps: f32,

    full_search_prob: f32,
    full_iterations: u32,
    part_iterations: f32,

    exploration_weight: f32,
    max_game_length: u32,
}

#[derive(Debug, Serialize)]
enum Update {
    FileDone,
    ThroughputUpdate {
        games_per_sec: f32,
        moves_per_sec: f32,
        cache_hit_rate: f32,
        new_evals_per_sec: f32,
    },
}

struct State {
    devices: Vec<DeviceState>,
}

struct DeviceState {
    device: Device,
    threads: Vec<Sender<Command>>,
}

fn start_collector_thread(startup_settings: StartupSettings) -> (JoinHandle<()>, Sender<Command>) {
    let (sender, receiver) = channel::unbounded();
    let handle = std::thread::spawn(|| {});
    (handle, sender)
}

fn start_generator_thread(settings: Settings) -> (JoinHandle<()>, Sender<Command>) {
    todo!()
}

fn main() -> anyhow::Result<()> {
    println!("{}", serde_json::to_string(&Command::Stop).unwrap());

    let args = std::env::args().collect_vec();
    assert_eq!(2, args.len(), "expected one argument, the output folder");

    let startup_settings = StartupSettings {
        output_folder: args[1].clone(),
    };

    println!("Startup settings: {:#?}", startup_settings);

    println!("Waiting for connection");
    let (mut stream, address) = TcpListener::bind("127.0.0.1:8668")?.accept()?;
    let mut reader = BufReader::new(stream.try_clone()?);
    println!("Accepted connection from {}", address);

    let (collector_handle, stop_collector) = start_collector_thread(startup_settings);

    let mut state = State {
        devices: all_cuda_devices().into_iter().map(|device| {
            DeviceState { device, threads: vec![] }
        }).collect(),
    };

    loop {
        let mut buf = vec![];
        reader.read_until(b'\n', &mut buf)?;
        let string = String::from_utf8(buf)?;
        println!("Received string '{}'", string);
        let command = serde_json::from_str::<Command>(&string)?;

        match command {
            Command::Stop => {
                println!("Received stop command");
                stop_collector.send(Command::Stop)?;
                break;
            }
            Command::NewSettings { settings } => {
                println!("Received new settings: {:#?}", settings);

                let new_thread_count = settings.threads_per_device as usize;

                for device in &mut state.devices {
                    let prev_thread_count = device.threads.len();
                    let kept_thread_count = min(new_thread_count, prev_thread_count);

                    // stop & remove extra threads
                    for sender in device.threads.drain(kept_thread_count..) {
                        sender.send(Command::Stop)?;
                    }

                    // update settings for threads that will stay alive
                    for sender in &device.threads {
                        sender.send(Command::NewSettings { settings: settings.clone() }).unwrap();
                    }

                    // start new threads
                    device.threads.extend(
                        (0..(new_thread_count - kept_thread_count))
                            .map(|_| start_generator_thread(settings.clone()).1)
                    )
                }
            }
        }

        let message = Update::FileDone;
        stream.write_all(serde_json::to_string(&message)?.as_bytes())?;
    }

    collector_handle.join().expect("Failed to join collector thread");

    Ok(())
}
