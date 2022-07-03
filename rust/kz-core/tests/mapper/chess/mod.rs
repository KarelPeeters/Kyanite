use kz_core::mapping::chess::generate_all_flat_moves_pov;

mod pairs;
mod random;

#[test]
fn flat_gen() {
    // the asserts are already in the function itself
    let mut moves = generate_all_flat_moves_pov();

    assert_eq!(moves.len(), 1880);

    // check for duplicates
    moves.sort();
    moves.dedup();
    assert_eq!(moves.len(), 1880);
}
