use std::fs::File;
use std::io::Read;
use std::path::Path;

use board_game::games::chess::{ChessBoard, Rules};
use bzip2::read::BzDecoder;
use crossbeam::channel::{Receiver, Sender};
use tar::Archive;

use alpha_zero::mapping::binary_output::BinaryOutput;
use alpha_zero::mapping::BoardMapper;
use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::mapping::pgn_to_bin::pgn_to_bin;

fn main() {
    let input = "../data/pgn-games/ccrl-pgn.tar.bz2";
    let output = "../data/pgn-games/";

    let rules = Rules::ccrl();
    let mapper = ChessStdMapper;

    convert(
        rules,
        mapper,
        input,
        output,
        true,
        None,
        None,
    );
}

fn convert(
    rules: Rules,
    mapper: impl BoardMapper<ChessBoard>,
    input_path: &str,
    output_folder: &str,
    skip_existing: bool,
    max_entries: Option<usize>,
    max_games_per_entry: Option<usize>,
) {
    std::fs::create_dir_all(output_folder)
        .expect("Failed to create output folder");

    let thread_count = 4;
    let (sender, receiver) = crossbeam::channel::bounded(thread_count);

    crossbeam::scope(|s| {
        s.spawn(|_| loader_main(input_path, &sender, max_entries));

        for _ in 0..thread_count {
            s.spawn(|_| mapper_main(output_folder, &receiver, rules, mapper, skip_existing, max_games_per_entry));
        }
    }).unwrap();
}

fn loader_main(input_path: &str, sender: &Sender<(String, Vec<u8>)>, max_entries: Option<usize>) {
    let input = File::open(input_path).expect("Failed to open input file");
    let mut archive = Archive::new(BzDecoder::new(input));

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
    output_folder: &str,
    receiver: &Receiver<(String, Vec<u8>)>,
    rules: Rules,
    mapper: impl BoardMapper<ChessBoard>,
    skip_existing: bool,
    max_games_per_entry: Option<usize>,
) {
    loop {
        let (path, data) = receiver.recv().unwrap();

        let mut output_path = Path::new(output_folder).join(&path);
        output_path.set_extension("");

        let parent = output_path.parent()
            .expect("No parent folder for file");
        std::fs::create_dir_all(parent)
            .expect("Error while creating output dirs");

        let output_path_json = output_path.with_extension(".json");

        if skip_existing && std::fs::metadata(&output_path_json).is_ok() {
            println!("Skipping {:?} because it already exists", path);
            return;
        }

        println!("Mapping {:?} to {:?}", path, output_path);

        let mut binary_output = BinaryOutput::new(&output_path, "chess", mapper)
            .expect("Error white opening output files");
        pgn_to_bin(rules, &*data, &mut binary_output, max_games_per_entry)
            .expect("Error while writing to binary");
        binary_output.finish()
            .expect("Error while finishing output");

        println!("Finished mapping {:?} to {:?}", path, output_path);
    }
}
