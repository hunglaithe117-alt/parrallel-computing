//! Benchmark `vector_add`: `out[i] = a[i] + b[i]`.
//!
//! Element-wise kernel: only the write loop is timed; the checksum runs in a
//! separate untimed pass over the output buffer (BENCH_CONTRACT.md v2).

use std::hint::black_box;

use simd_lab::{
    bench_report, elementwise_checksum, fill_inputs, n_from_args, print_context, report_agreement,
    time_median_unit, vector_add_autovec, vector_add_avx2_fma, vector_add_rayon, vector_add_scalar,
    verbose,
};

const DEFAULT_N: usize = 8_000_000;
const EXAMPLE: &str = "vector_add";

fn main() {
    let n = n_from_args(DEFAULT_N);
    let verbose = verbose();
    print_context();

    let (a, b) = fill_inputs(n);
    let mut out = vec![0.0_f32; n];

    let timing = time_median_unit(|| {
        vector_add_scalar(&a, &b, &mut out);
        let _ = black_box(out.as_slice());
    });
    let scalar = bench_report(EXAMPLE, "scalar", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        vector_add_autovec(&a, &b, &mut out);
        let _ = black_box(out.as_slice());
    });
    let autovec = bench_report(EXAMPLE, "autovec", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        vector_add_avx2_fma(&a, &b, &mut out);
        let _ = black_box(out.as_slice());
    });
    let avx2 = bench_report(EXAMPLE, "avx2_fma", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        vector_add_rayon(&a, &b, &mut out);
        let _ = black_box(out.as_slice());
    });
    let rayon = bench_report(EXAMPLE, "rayon", n, timing, elementwise_checksum(&out));

    if verbose {
        report_agreement(EXAMPLE, scalar, "autovec", autovec);
        report_agreement(EXAMPLE, scalar, "avx2_fma", avx2);
        report_agreement(EXAMPLE, scalar, "rayon", rayon);
    }
}
