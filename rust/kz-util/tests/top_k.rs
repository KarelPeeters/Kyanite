use itertools::Itertools;
use kz_util::{top_k_indices_sorted, PrintThroughput};
use rand::{thread_rng, Rng};

#[test]
#[ignore]
fn top_k_bench() {
    let n = 1880;
    let k = 100;

    let mut rng = thread_rng();

    let x = (0..n).map(|_| rng.gen::<f32>()).collect_vec();

    let mut tp = PrintThroughput::new("top_k");
    let mut total = 0;

    for _ in 0..1_000_000 {
        let r = top_k_indices_sorted(&x, k);
        total += r[0];
        tp.update_delta(1);
    }

    println!("{}", total);
}
