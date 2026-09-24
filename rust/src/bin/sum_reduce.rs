//! Benchmark `sum_reduce`: `sum(x[i])`.

use simd_lab::{
    bench_variant, fill_inputs, n_from_args, print_context, report_agreement, sum_reduce_autovec,
    sum_reduce_avx2_fma, sum_reduce_rayon, sum_reduce_scalar, verbose,
};

const DEFAULT_N: usize = 8_000_000;
const EXAMPLE: &str = "sum_reduce";

fn main() {
    let n = n_from_args(DEFAULT_N);
    let verbose = verbose();
    print_context();

    let (x, _) = fill_inputs(n);

    let scalar = bench_variant(EXAMPLE, "scalar", n, || sum_reduce_scalar(&x));
    let autovec = bench_variant(EXAMPLE, "autovec", n, || sum_reduce_autovec(&x));
    let avx2 = bench_variant(EXAMPLE, "avx2_fma", n, || sum_reduce_avx2_fma(&x));
    let rayon = bench_variant(EXAMPLE, "rayon", n, || sum_reduce_rayon(&x));

    if verbose {
        report_agreement(EXAMPLE, scalar, "autovec", autovec);
        report_agreement(EXAMPLE, scalar, "avx2_fma", avx2);
        report_agreement(EXAMPLE, scalar, "rayon", rayon);
    }
}
