use itertools::Itertools;
use sttt::board::{Coord, Symmetry};

fn main() {
    println!("SYMMETRY_INDICES_YX = [");
    for sym in Symmetry::all() {
        let vec = Coord::all_yx()
            .map(|c| sym.map_coord(c).yx())
            .collect_vec();
        println!("  {:?}, ", vec);
    }
    println!("]");
}
