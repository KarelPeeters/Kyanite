use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use bzip2::read::BzDecoder;
use clap::Parser;
use memmap::Mmap;

use alpha_zero::convert::pgn_archive_to_bin::pgn_archive_to_bin;
use alpha_zero::convert::pgn_to_bin::{append_pgn_to_bin, Filter};
use alpha_zero::mapping::binary_output::BinaryOutput;
use alpha_zero::mapping::chess::ChessStdMapper;

#[derive(Debug, clap::Parser)]
struct Opts {
    #[clap(long)]
    tc: Vec<String>,

    #[clap(long)]
    min_elo: Option<u32>,
    #[clap(long)]
    max_elo: Option<u32>,

    #[clap(long)]
    skip_existing: bool,
    #[clap(long)]
    thread_count: Option<usize>,
    #[clap(long)]
    max_games: Option<u32>,

    input: PathBuf,
}

fn main() {
    let opts: Opts = Opts::parse();
    println!("Using options {:#?}", opts);

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

    let mapper = ChessStdMapper;

    let filter = Filter {
        min_elo: opts.max_elo,
        max_elo: opts.min_elo,
        allowed_time_controls: (!opts.tc.is_empty()).then(|| opts.tc.clone()),
    };

    let ext = path.extension().and_then(|e| e.to_str());
    let thread_count = opts.thread_count.unwrap_or(4);

    match ext {
        Some("tar") => {
            let output_folder = path.file_stem().unwrap();
            println!("Writing to output folder {:?}", output_folder);
            pgn_archive_to_bin(
                mapper, input, output_folder,
                thread_count, opts.skip_existing, &filter, None, None,
            )
        }
        Some("pgn") => {
            let input_file = File::open(path).expect("Failed to open input file");
            let input = unsafe { Mmap::map(&input_file) }.expect("Failed to memmap input file");

            let mut binary_output = BinaryOutput::new(path.with_extension(""), "chess", mapper).unwrap();
            append_pgn_to_bin(&input, &mut binary_output, &filter, opts.max_games, true).unwrap();
            binary_output.finish().unwrap();
        }
        _ => panic!("Unexpected extension in (sub) path  {:?}", path),
    }
}
