use std::fs::File;
use alpha_zero::mapping::pgn_to_bin::pgn_to_bin;
use alpha_zero::mapping::chess::ChessStdMapper;
use tar::Archive;
use bzip2::read::BzDecoder;
use alpha_zero::mapping::binary_output::BinaryOutput;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::path::Path;

fn main() {
    let input = "../data/pgn-games/ccrl-pgn.tar.bz2";
    let output = "../data/pgn-games/";
    convert(input, output, None, None);
}

fn convert(input_path: &str, output_folder: &str, max_entries: Option<usize>, max_games_per_entry: Option<usize>) {
    let input = File::open(input_path).expect("Failed to open input file");
    let mut archive = Archive::new(BzDecoder::new(input));

    let entries = archive.entries()
        .expect("Corrupt archive")
        .take(max_entries.unwrap_or(usize::MAX));

    for entry in entries {
        let entry = entry.expect("Corrupt entry");
        let entry_path = entry.path()
            .expect("Corrupt path");

        if entry_path.to_string_lossy().contains("train") { continue; }

        let mut output_path = Path::new(output_folder).join(entry_path);
        output_path.set_extension("bin.gz");
        std::fs::create_dir_all(output_path.parent().unwrap())
            .expect("Failed to create output folder");

        println!("Mapping {:?} to {:?}", entry.path().unwrap().as_ref(), output_path);

        let output = File::create(&output_path).expect("Failed to create output file");
        let mut binary_output = BinaryOutput::new(ChessStdMapper, GzEncoder::new(output, Compression::fast()));

        pgn_to_bin(entry, &mut binary_output, max_games_per_entry)
            .expect("Error white writing to binary");
    }
}