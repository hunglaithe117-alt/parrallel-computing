//! Benchmark `dot_product`: `sum(a[i] * b[i])`.

use simd_lab::{
    bench_variant, dot_product_autovec, dot_product_avx2_fma, dot_product_rayon,
    dot_product_scalar, fill_inputs, n_from_args, print_context, report_agreement, verbose,
};

const DEFAULT_N: usize = 8_000_000;
const EXAMPLE: &str = "dot_product";

fn main() {
    let n = n_from_args(DEFAULT_N);
    let verbose = verbose();
    print_context();

    let (a, b) = fill_inputs(n);

    let scalar = bench_variant(EXAMPLE, "scalar", n, || dot_product_scalar(&a, &b));
    let autovec = bench_variant(EXAMPLE, "autovec", n, || dot_product_autovec(&a, &b));
    let avx2 = bench_variant(EXAMPLE, "avx2_fma", n, || dot_product_avx2_fma(&a, &b));
    let rayon = bench_variant(EXAMPLE, "rayon", n, || dot_product_rayon(&a, &b));

    if verbose {
        report_agreement(EXAMPLE, scalar, "autovec", autovec);
        report_agreement(EXAMPLE, scalar, "avx2_fma", avx2);
        report_agreement(EXAMPLE, scalar, "rayon", rayon);
    }
}
