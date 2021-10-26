use std::io::Read;
use std::path::Path;

use board_game::games::chess::{ChessBoard, Rules};
use crossbeam::channel::{Receiver, Sender};
use tar::Archive;

use crate::convert::pgn_to_bin::append_pgn_to_bin;
use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;

pub fn pgn_archive_to_bin(
    rules: Rules,
    mapper: impl BoardMapper<ChessBoard>,
    input: impl Read + Send,
    output_folder: impl AsRef<Path>,
    thread_count: usize,
    skip_existing: bool,
    min_elo: Option<u32>,
    max_elo: Option<u32>,
    max_entries: Option<usize>,
    max_games_per_entry: Option<u32>,
) {
    let output_folder = output_folder.as_ref();
    std::fs::create_dir_all(output_folder)
        .expect("Failed to create output folder");

    let (sender, receiver) = crossbeam::channel::bounded(thread_count);

    crossbeam::scope(|s| {
        s.spawn(|_| loader_main(input, sender, max_entries));

        for _ in 0..thread_count {
            s.spawn(|_| mapper_main(output_folder, &receiver, rules, mapper, min_elo, max_elo, skip_existing, max_games_per_entry));
        }
    }).unwrap();
}

fn loader_main(input: impl Read, sender: Sender<(String, Vec<u8>)>, max_entries: Option<usize>) {
    let mut archive = Archive::new(input);

    let max_entries = max_entries.unwrap_or(usize::MAX);

    for entry in archive.entries().unwrap().take(max_entries) {
        let mut entry = entry.unwrap();

        let path = entry.path().as_ref().unwrap().to_str().unwrap().to_owned();

        println!("Reading entry {}", path);
        let mut data = vec![];
        entry.read_to_end(&mut data).unwrap();
        println!("Finished reading entry {}", path);

        sender.send((path.to_owned(), data)).unwrap();
    }
}

fn mapper_main(
    output_folder: impl AsRef<Path>,
    receiver: &Receiver<(String, Vec<u8>)>,
    rules: Rules,
    mapper: impl BoardMapper<ChessBoard>,
    min_elo: Option<u32>,
    max_elo: Option<u32>,
    skip_existing: bool,
    max_games_per_entry: Option<u32>,
) {
    let output_folder = output_folder.as_ref();

    loop {
        let (path, data) = match receiver.recv() {
            Ok(content) => content,
            Err(_) => break,
        };

        let mut output_path = output_folder.join(&path);
        output_path.set_extension("");

        let parent = output_path.parent()
            .expect("No parent folder for file");
        std::fs::create_dir_all(parent)
            .expect("Error while creating output dirs");

        let output_path_json = output_path.with_extension("json");

        if skip_existing && std::fs::metadata(&output_path_json).is_ok() {
            println!("Skipping {:?} because it already exists", path);
            continue;
        }

        println!("Mapping {:?} to {:?}", path, output_path);

        let mut binary_output = BinaryOutput::new(&output_path, "chess", mapper)
            .expect("Error white opening output files");
        append_pgn_to_bin(rules, &*data, &mut binary_output, min_elo, max_elo, max_games_per_entry, false)
            .expect("Error while writing to binary");
        binary_output.finish()
            .expect("Error while finishing output");

        println!("Finished mapping {:?} to {:?}", path, output_path);
    }
}