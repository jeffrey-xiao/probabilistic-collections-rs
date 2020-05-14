use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use probabilistic_collections::quotient::QuotientFilter;

fn bench_insert(c: &mut Criterion) {
    let mut initial_items = 0;
    while initial_items < 1024 - 32 {
        c.bench_function(&format!("bench insert {}", initial_items), |b| {
            b.iter_batched_ref(
                || {
                    let mut filter = QuotientFilter::<u32>::from_fpp(1024, 0.01);
                    for i in 0..initial_items {
                        filter.insert(&i);
                    }
                    filter
                },
                |filter| filter.insert(&0xDEADBEEF),
                BatchSize::PerIteration,
            )
        });
        initial_items += 32;
    }
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
