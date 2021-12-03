use alpha_zero::mapping::chess::generate_all_flat_moves_pov;

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