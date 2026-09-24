//! Benchmark `mandelbrot`: sum of escape-iteration counts over a grid.
//!
//! `BENCH_N` is interpreted as a requested pixel count; the width is fixed at
//! [`WIDTH`] and the height is derived, so the reported `n` is `WIDTH * height`.

use simd_lab::{
    bench_variant, mandelbrot_autovec, mandelbrot_avx2_fma, mandelbrot_rayon, mandelbrot_scalar,
    n_from_args, print_context, report_agreement, verbose,
};

const WIDTH: usize = 1024;
const DEFAULT_N: usize = 1024 * 768;
const MAX_ITER: usize = 100;
const EXAMPLE: &str = "mandelbrot";

fn main() {
    let requested = n_from_args(DEFAULT_N);
    let height = (requested / WIDTH).max(1);
    let pixels = WIDTH * height;
    let verbose = verbose();
    print_context();

    let scalar = bench_variant(EXAMPLE, "scalar", pixels, || {
        mandelbrot_scalar(WIDTH, height, MAX_ITER)
    });
    let autovec = bench_variant(EXAMPLE, "autovec", pixels, || {
        mandelbrot_autovec(WIDTH, height, MAX_ITER)
    });
    let avx2 = bench_variant(EXAMPLE, "avx2_fma", pixels, || {
        mandelbrot_avx2_fma(WIDTH, height, MAX_ITER)
    });
    let rayon = bench_variant(EXAMPLE, "rayon", pixels, || {
        mandelbrot_rayon(WIDTH, height, MAX_ITER)
    });

    if verbose {
        report_agreement(EXAMPLE, scalar, "autovec", autovec);
        report_agreement(EXAMPLE, scalar, "avx2_fma", avx2);
        report_agreement(EXAMPLE, scalar, "rayon", rayon);
    }
}
