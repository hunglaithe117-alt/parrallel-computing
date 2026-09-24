//! Benchmark `saxpy`: `out[i] = alpha * x[i] + y[i]`.
//!
//! Element-wise, out-of-place kernel: only the write loop is timed; the
//! checksum runs in a separate untimed pass over the output buffer
//! (BENCH_CONTRACT.md v2). Because `out` is distinct from `x`/`y`, repeated
//! timed reps read pristine inputs and the checksum is `REPS`-invariant.

use std::hint::black_box;

use simd_lab::{
    bench_report, elementwise_checksum, fill_inputs, n_from_args, print_context, report_agreement,
    saxpy_autovec, saxpy_avx2_fma, saxpy_rayon, saxpy_scalar, time_median_unit, verbose,
};

const DEFAULT_N: usize = 8_000_000;
const EXAMPLE: &str = "saxpy";
const ALPHA: f32 = 2.5;

fn main() {
    let n = n_from_args(DEFAULT_N);
    let verbose = verbose();
    print_context();

    let (x, y) = fill_inputs(n);
    let mut out = vec![0.0_f32; n];

    let timing = time_median_unit(|| {
        saxpy_scalar(ALPHA, &x, &y, &mut out);
        let _ = black_box(out.as_slice());
    });
    let scalar = bench_report(EXAMPLE, "scalar", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        saxpy_autovec(ALPHA, &x, &y, &mut out);
        let _ = black_box(out.as_slice());
    });
    let autovec = bench_report(EXAMPLE, "autovec", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        saxpy_avx2_fma(ALPHA, &x, &y, &mut out);
        let _ = black_box(out.as_slice());
    });
    let avx2 = bench_report(EXAMPLE, "avx2_fma", n, timing, elementwise_checksum(&out));

    let timing = time_median_unit(|| {
        saxpy_rayon(ALPHA, &x, &y, &mut out);
        let _ = black_box(out.as_slice());
    });
    let rayon = bench_report(EXAMPLE, "rayon", n, timing, elementwise_checksum(&out));

    if verbose {
        report_agreement(EXAMPLE, scalar, "autovec", autovec);
        report_agreement(EXAMPLE, scalar, "avx2_fma", avx2);
        report_agreement(EXAMPLE, scalar, "rayon", rayon);
    }
}
