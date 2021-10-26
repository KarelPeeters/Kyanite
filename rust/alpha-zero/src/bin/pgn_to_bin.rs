use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use board_game::games::chess::Rules;
use bzip2::read::BzDecoder;
use clap::Parser;

use alpha_zero::convert::pgn_archive_to_bin::pgn_archive_to_bin;
use alpha_zero::convert::pgn_to_bin::append_pgn_to_bin;
use alpha_zero::mapping::binary_output::BinaryOutput;
use alpha_zero::mapping::chess::ChessStdMapper;

#[derive(clap::Parser)]
struct Opts {
    #[clap(long, possible_values = ["ccrl", "default"])]
    rules: String,

    #[clap(long)]
    min_elo: Option<u32>,
    #[clap(long)]
    max_elo: Option<u32>,

    #[clap(long)]
    skip_existing: bool,
    #[clap(long)]
    thread_count: Option<usize>,

    input: PathBuf,
    output: Option<PathBuf>,
}

fn main() {
    let opts: Opts = Opts::parse();

    let input = File::open(&opts.input).expect("Failed to open input file");

    let ext = opts.input.extension().and_then(|e| e.to_str());
    if ext == Some("bz2") {
        println!("Reading compressed file");
        main_dispatch(&opts, &opts.input.with_extension(""), BzDecoder::new(input));
    } else {
        main_dispatch(&opts, &opts.input, input)
    }
}

fn main_dispatch(opts: &Opts, path: &Path, input: impl Read + Send) {
    println!("Input {:?}", path);

    let rules = match &*opts.rules {
        "ccrl" => Rules::ccrl(),
        "default" => Rules::default(),
        _ => unreachable!(),
    };
    let mapper = ChessStdMapper;

    let ext = path.extension().and_then(|e| e.to_str());
    let thread_count = opts.thread_count.unwrap_or(4);

    match ext {
        Some("tar") => {
            let output_folder = path.file_stem().unwrap();
            println!("Writing to output folder {:?}", output_folder);
            pgn_archive_to_bin(
                rules, mapper, input, output_folder,
                thread_count, opts.skip_existing, opts.min_elo, opts.max_elo, None, None,
            )
        }
        Some("pgn") => {
            let mut binary_output = BinaryOutput::new(path.with_extension(""), "chess", mapper).unwrap();
            append_pgn_to_bin(rules, input, &mut binary_output, opts.min_elo, opts.max_elo, Some(1_000_000), true).unwrap();
            binary_output.finish().unwrap();
        }
        _ => panic!("Unexpected extension in (sub) path  {:?}", path),
    }
}
