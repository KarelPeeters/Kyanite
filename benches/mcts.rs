use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sttt::mcts::old_move_mcts;
use sttt::board::Board;
use rand::thread_rng;

fn foo(c: &mut Criterion) {
    c.bench_function("mcts 1000", |b| {
        b.iter(|| old_move_mcts(&Board::new(), 100_000, &mut thread_rng(), false))
    });
}

criterion_group!(benches, foo);
criterion_main!(benches);