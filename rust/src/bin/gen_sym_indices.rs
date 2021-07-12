use itertools::Itertools;
use sttt::board::{Coord, Symmetry};

fn main() {
    println!("SYMMETRY_INDICES_O = [");
    for sym in Symmetry::all() {
        let vec = Coord::all()
            .map(|c| sym.map_coord(c).o())
            .collect_vec();
        println!("  {:?}, ", vec);
    }
    println!("]");

    println!("SYMMETRY_INDICES_OO = [");
    for sym in Symmetry::all() {
        let vec = (0..9)
            .map(|oo| sym.map_oo(oo))
            .collect_vec();
        println!("  {:?}, ", vec);
    }
    println!("]");
}
