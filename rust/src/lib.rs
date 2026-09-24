//! SIMD and parallel-computing kernels plus the benchmark harness for this lab.
//!
//! Timing follows `docs/BENCH_CONTRACT.md` (v2): one untimed warmup, then the
//! median of `BENCH_REPS` (default 7) `Instant`-timed repetitions. Element-wise
//! kernels time only the write loop and compute their checksum in a separate
//! untimed pass; reduction kernels time the reduction itself and use its result
//! as the checksum. Every result is routed through [`std::hint::black_box`] so
//! dead-code elimination cannot remove the measured work.
//!
//! Four implementation strategies are offered per kernel:
//!
//! * `scalar`    - the straightforward indexed loop, the reference variant.
//! * `autovec`   - safe iterator code that LLVM auto-vectorizes under
//!   `-C target-cpu=native`.
//! * `avx2_fma`  - explicit 256-bit intrinsics in an `unsafe fn` marked
//!   `#[target_feature(enable = "avx2,fma")]`, reached only through a runtime
//!   [`is_x86_feature_detected!`] guard with a scalar fallback.
//! * `rayon`     - data-parallel split of the same kernel across the thread pool.
//!
//! `scalar` is a checksum reference, not a promise of scalar machine code.
//! `black_box(len)` only hides the trip count from constant folding; it does not
//! stop LLVM's loop vectorizer. An element-wise loop with no reduction is still
//! vectorized. The real barrier is the non-reassociable strict-`f64` reduction
//! in `dot_product`/`sum_reduce`, which LLVM will not reorder without fast-math.
//!
//! No module in this crate uses `std::simd`, which is unstable (nightly-only).

use std::collections::HashSet;
use std::hint::black_box;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Relative tolerance used when comparing floating-point checksums across
/// variants, per `docs/BENCH_CONTRACT.md`.
pub const REL_TOL: f64 = 1e-4;

/// Default number of timed repetitions when `BENCH_REPS` is unset.
const DEFAULT_REPS: usize = 7;

/// Returns the configured repetition count (`BENCH_REPS`, default 7).
pub fn reps() -> usize {
    std::env::var("BENCH_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_REPS)
}

/// Returns `true` when `--verbose` appears on the command line.
pub fn verbose() -> bool {
    std::env::args().any(|arg| arg == "--verbose")
}

/// Resolves the element count from `--n <count>` / `--n=<count>`, else the
/// `BENCH_N` environment variable, else `default`.
///
/// Per BENCH_CONTRACT.md v2 measurement rule 8, a non-positive or malformed
/// count is rejected with a clear error on stderr and exit status 2.
pub fn n_from_args(default: usize) -> usize {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--n" {
            let value = args.next().unwrap_or_else(|| die("--n requires a value"));
            return parse_n(&value);
        } else if let Some(value) = arg.strip_prefix("--n=") {
            return parse_n(value);
        }
    }
    match std::env::var("BENCH_N") {
        Ok(value) => parse_n(&value),
        Err(_) => default,
    }
}

/// Parses an explicit element count, rejecting `n <= 0` and non-numeric input.
fn parse_n(raw: &str) -> usize {
    match raw.trim().parse::<i64>() {
        Ok(value) if value > 0 => value as usize,
        Ok(value) => die(&format!("n must be > 0, got {value}")),
        Err(_) => die(&format!("invalid n value: {raw:?}")),
    }
}

/// Prints a benchmark setup error to stderr and exits with status 2.
fn die(message: &str) -> ! {
    eprintln!("[rust] ERROR: {message}");
    std::process::exit(2);
}

/// Emits one line in the frozen wire format:
/// `BENCH,<example>,<variant>,<n>,<ms>,<checksum>`.
///
/// `ms` is printed with four decimals; `checksum` with six.
pub fn emit_bench(example: &str, variant: &str, n: usize, ms: f64, checksum: f64) {
    println!("BENCH,{example},{variant},{n},{ms:.4},{checksum:.6}");
}

/// Timing statistics for one variant. `median_ms` is the value reported in the
/// `BENCH,` line; `min_ms` is the dispersion figure required by contract rule 7.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// Median of the timed samples, in milliseconds.
    pub median_ms: f64,
    /// Fastest timed sample, in milliseconds.
    pub min_ms: f64,
}

/// Sorts samples and returns the median (average of the two middle samples when
/// `REPS` is even) plus the minimum, per contract rule 2.
fn summarize(samples: &mut [f64]) -> Timing {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = samples.len();
    let median_ms = if len % 2 == 1 {
        samples[len / 2]
    } else {
        (samples[len / 2 - 1] + samples[len / 2]) / 2.0
    };
    Timing {
        median_ms,
        min_ms: samples[0],
    }
}

/// Times a kernel closure that returns its checksum (reduction kernels), with
/// one untimed warmup. The reduction runs inside the timed region; its result is
/// the checksum.
pub fn time_median<F>(mut f: F) -> (Timing, f64)
where
    F: FnMut() -> f64,
{
    let _ = black_box(f());

    let reps = reps();
    let mut samples = Vec::with_capacity(reps);
    let mut checksum = 0.0_f64;
    for _ in 0..reps {
        let start = Instant::now();
        checksum = black_box(f());
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }

    (summarize(&mut samples), checksum)
}

/// Times an element-wise kernel closure that returns `()`, with one untimed
/// warmup. The checksum is computed separately by the caller, untimed, after
/// this returns.
pub fn time_median_unit<F>(mut f: F) -> Timing
where
    F: FnMut(),
{
    f();

    let reps = reps();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }

    summarize(&mut samples)
}

/// Emits one variant's `BENCH,` line to stdout, then its dispersion summary
/// (`median`, `min`, `checksum`) to stderr **unconditionally**, as required by
/// BENCH_CONTRACT.md rules 5 and 7. Returns the checksum for cross-checking.
pub fn bench_report(example: &str, variant: &str, n: usize, timing: Timing, checksum: f64) -> f64 {
    emit_bench(example, variant, n, timing.median_ms, checksum);
    eprintln!(
        "[rust] {example} {variant}: n={n} median={:.4} ms  min={:.4} ms  checksum={checksum:.6}",
        timing.median_ms, timing.min_ms
    );
    checksum
}

/// Runs a reduction/output variant whose closure produces the checksum inside
/// the timed region.
pub fn bench_variant<F>(example: &str, variant: &str, n: usize, f: F) -> f64
where
    F: FnMut() -> f64,
{
    let (timing, checksum) = time_median(f);
    bench_report(example, variant, n, timing, checksum)
}

/// True relative tolerance comparison per contract v2: denominator
/// `max(|reference|, 1e-12)`.
pub fn checks_agree(reference: f64, candidate: f64) -> bool {
    let denom = reference.abs().max(1e-12);
    (candidate - reference).abs() / denom <= REL_TOL
}

/// Prints an `ok`/`FAIL` correctness line for one variant versus the scalar
/// reference. Verbose mode only; stderr, never stdout.
pub fn report_agreement(example: &str, reference: f64, variant: &str, value: f64) {
    let status = if checks_agree(reference, value) {
        "ok"
    } else {
        "FAIL"
    };
    let denom = reference.abs().max(1e-12);
    eprintln!(
        "[rust] {example}: {variant} vs scalar -> {status} (rel diff {:.3e})",
        (value - reference).abs() / denom
    );
}

/// Build-time context is available through this module-level accessor so the
/// library does not depend on `env!` in generic code.
fn rustc_version() -> &'static str {
    option_env!("RUSTC_VERSION").unwrap_or("unknown")
}

fn build_rustflags() -> &'static str {
    option_env!("BUILD_RUSTFLAGS").unwrap_or("")
}

/// Prints compiler, SIMD, and thread context once per process, to stderr.
///
/// Satisfies BENCH_CONTRACT.md rule 5 without polluting the `BENCH,` stdout
/// stream.
pub fn print_context() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let threads = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        let avx2 = has_avx2();
        let fma = has_fma();
        eprintln!(
            "[rust] {} {} | {} | avx2={} fma={} | threads={}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            rustc_version(),
            avx2,
            fma,
            threads,
        );
        let flags = build_rustflags();
        eprintln!(
            "[rust] RUSTFLAGS={}",
            if flags.is_empty() { "(unset)" } else { flags }
        );
    });
}

/// `true` when the CPU reports AVX2 (x86-64 only; `false` elsewhere).
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `true` when the CPU reports FMA (x86-64 only; `false` elsewhere).
pub fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Announces once per `example` which path the `avx2_fma` variant took.
///
/// Printed to stderr so the fallback is auditable rather than silent.
fn announce(example: &'static str, available: bool) {
    static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = seen.lock().expect("announce mutex poisoned");
    if guard.insert(example) {
        if available {
            eprintln!(
                "[rust] {example}: avx2_fma -> AVX2+FMA detected, using core::arch intrinsics"
            );
        } else {
            eprintln!("[rust] {example}: avx2_fma -> AVX2/FMA unavailable, falling back to scalar");
        }
    }
}

// ---------------------------------------------------------------------------
// Shared input generation
// ---------------------------------------------------------------------------

/// Deterministic, strictly positive inputs shared by every variant so the
/// checksums are well conditioned for relative comparison.
pub fn fill_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push(1.0 + (i % 17) as f32 * 0.031_25);
        b.push(0.5 + (i % 11) as f32 * 0.0625);
    }
    (a, b)
}

/// Sum of a slice as `f64`, the shared checksum shape for reductions.
fn sum_f64(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().sum()
}

/// Untimed checksum pass for element-wise kernels: sums the output buffer.
///
/// `black_box` makes the whole buffer observable, so the stores performed by
/// the timed write loop cannot be eliminated.
pub fn elementwise_checksum(out: &[f32]) -> f64 {
    sum_f64(black_box(out).iter().map(|&value| f64::from(value)))
}

// ---------------------------------------------------------------------------
// vector_add: out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------

/// Sequential reference element-wise loop. `black_box(n)` only hides the trip
/// count from constant folding; it does **not** block the loop vectorizer, so
/// LLVM still vectorizes this loop (see README). The timed region is the write
/// loop only; the checksum is computed separately.
pub fn vector_add_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = black_box(a.len().min(b.len()).min(out.len()));
    let mut i = 0_usize;
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Safe iterator form that LLVM auto-vectorizes under `-C target-cpu=native`.
pub fn vector_add_autovec(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x + y;
    }
}

/// AVX2 + FMA implementation; only call through [`vector_add_avx2_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vector_add_avx2_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::*;

    unsafe {
        let n = a.len().min(b.len()).min(out.len());
        let mut i = 0_usize;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
            i += 8;
        }
        while i < n {
            out[i] = a[i] + b[i];
            i += 1;
        }
    }
}

/// Runtime-dispatched `avx2_fma` variant of [`vector_add_scalar`].
pub fn vector_add_avx2_fma(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            announce("vector_add", true);
            unsafe { vector_add_avx2_impl(a, b, out) };
            return;
        }
    }
    announce("vector_add", false);
    vector_add_scalar(a, b, out);
}

/// Rayon data-parallel variant of [`vector_add_scalar`].
pub fn vector_add_rayon(a: &[f32], b: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;

    out.par_iter_mut()
        .zip(a.par_iter())
        .zip(b.par_iter())
        .for_each(|((o, &x), &y)| *o = x + y);
}

// ---------------------------------------------------------------------------
// saxpy: out[i] = alpha * x[i] + y[i]
// ---------------------------------------------------------------------------

/// Sequential reference element-wise loop (timed write loop only).
pub fn saxpy_scalar(alpha: f32, x: &[f32], y: &[f32], out: &mut [f32]) {
    let n = black_box(x.len().min(y.len()).min(out.len()));
    let mut i = 0_usize;
    while i < n {
        out[i] = alpha * x[i] + y[i];
        i += 1;
    }
}

/// Safe iterator form; `mul_add` lets LLVM emit a fused multiply-add under FMA.
pub fn saxpy_autovec(alpha: f32, x: &[f32], y: &[f32], out: &mut [f32]) {
    for ((o, &xv), &yv) in out.iter_mut().zip(x.iter()).zip(y.iter()) {
        *o = alpha.mul_add(xv, yv);
    }
}

/// AVX2 + FMA implementation; only call through [`saxpy_avx2_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn saxpy_avx2_impl(alpha: f32, x: &[f32], y: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::*;

    unsafe {
        let n = x.len().min(y.len()).min(out.len());
        let alpha_v = _mm256_set1_ps(alpha);
        let mut i = 0_usize;
        while i + 8 <= n {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vy = _mm256_loadu_ps(y.as_ptr().add(i));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(alpha_v, vx, vy));
            i += 8;
        }
        while i < n {
            out[i] = alpha.mul_add(x[i], y[i]);
            i += 1;
        }
    }
}

/// Runtime-dispatched `avx2_fma` variant of [`saxpy_scalar`].
pub fn saxpy_avx2_fma(alpha: f32, x: &[f32], y: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            announce("saxpy", true);
            unsafe { saxpy_avx2_impl(alpha, x, y, out) };
            return;
        }
    }
    announce("saxpy", false);
    saxpy_scalar(alpha, x, y, out);
}

/// Rayon data-parallel variant of [`saxpy_scalar`].
pub fn saxpy_rayon(alpha: f32, x: &[f32], y: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;

    out.par_iter_mut()
        .zip(x.par_iter())
        .zip(y.par_iter())
        .for_each(|((o, &xv), &yv)| *o = alpha.mul_add(xv, yv));
}

// ---------------------------------------------------------------------------
// dot_product: sum(a[i] * b[i])
// ---------------------------------------------------------------------------

/// Sequential reference reduction, accumulating in `f64`. `black_box(n)` only
/// hides the trip count from constant folding; the strict-`f64` sum is what
/// prevents vectorization (LLVM may not reassociate it). The reduction runs
/// inside the timed region and its result is the checksum.
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f64 {
    let n = black_box(a.len().min(b.len()));
    let mut sum = 0.0_f64;
    let mut i = 0_usize;
    while i < n {
        sum += f64::from(a[i]) * f64::from(b[i]);
        i += 1;
    }
    sum
}

/// Safe iterator reduction. LLVM does **not** auto-vectorize this strict-`f64`
/// sum: without fast-math it may not reassociate the reduction across lanes, so
/// this stays scalar (see README).
pub fn dot_product_autovec(a: &[f32], b: &[f32]) -> f64 {
    sum_f64(
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| f64::from(x) * f64::from(y)),
    )
}

/// AVX2 + FMA implementation with four `f64` accumulators; only call through
/// [`dot_product_avx2_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_impl(a: &[f32], b: &[f32]) -> f64 {
    use core::arch::x86_64::*;

    unsafe {
        let n = a.len().min(b.len());
        let mut i = 0_usize;
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();

        while i + 16 <= n {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));

            let a0lo = _mm256_cvtps_pd(_mm256_castps256_ps128(va0));
            let b0lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vb0));
            let a0hi = _mm256_cvtps_pd(_mm256_extractf128_ps(va0, 1));
            let b0hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vb0, 1));
            let a1lo = _mm256_cvtps_pd(_mm256_castps256_ps128(va1));
            let b1lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vb1));
            let a1hi = _mm256_cvtps_pd(_mm256_extractf128_ps(va1, 1));
            let b1hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vb1, 1));

            acc0 = _mm256_fmadd_pd(a0lo, b0lo, acc0);
            acc1 = _mm256_fmadd_pd(a0hi, b0hi, acc1);
            acc2 = _mm256_fmadd_pd(a1lo, b1lo, acc2);
            acc3 = _mm256_fmadd_pd(a1hi, b1hi, acc3);
            i += 16;
        }

        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(va));
            let blo = _mm256_cvtps_pd(_mm256_castps256_ps128(vb));
            let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1));
            let bhi = _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1));
            acc0 = _mm256_fmadd_pd(lo, blo, acc0);
            acc1 = _mm256_fmadd_pd(hi, bhi, acc1);
            i += 8;
        }

        let combined = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        let mut sum = hsum_pd(combined);
        while i < n {
            sum += f64::from(a[i]) * f64::from(b[i]);
            i += 1;
        }
        sum
    }
}

/// Runtime-dispatched `avx2_fma` variant of [`dot_product_scalar`].
pub fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            announce("dot_product", true);
            return unsafe { dot_product_avx2_impl(a, b) };
        }
    }
    announce("dot_product", false);
    dot_product_scalar(a, b)
}

/// Rayon data-parallel variant of [`dot_product_scalar`].
pub fn dot_product_rayon(a: &[f32], b: &[f32]) -> f64 {
    use rayon::prelude::*;

    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

// ---------------------------------------------------------------------------
// sum_reduce: sum(x[i])
// ---------------------------------------------------------------------------

/// Sequential reference reduction, accumulating in `f64`. `black_box(n)` only
/// hides the trip count from constant folding; the strict-`f64` sum is what
/// prevents vectorization (LLVM may not reassociate it). The reduction runs
/// inside the timed region and its result is the checksum.
pub fn sum_reduce_scalar(x: &[f32]) -> f64 {
    let n = black_box(x.len());
    let mut sum = 0.0_f64;
    let mut i = 0_usize;
    while i < n {
        sum += f64::from(x[i]);
        i += 1;
    }
    sum
}

/// Safe iterator reduction. LLVM does **not** auto-vectorize this strict-`f64`
/// sum: without fast-math it may not reassociate the reduction across lanes, so
/// this stays scalar (see README).
pub fn sum_reduce_autovec(x: &[f32]) -> f64 {
    sum_f64(x.iter().map(|&value| f64::from(value)))
}

/// AVX2 + FMA implementation with four `f64` accumulators; only call through
/// [`sum_reduce_avx2_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_reduce_avx2_impl(x: &[f32]) -> f64 {
    use core::arch::x86_64::*;

    unsafe {
        let n = x.len();
        let mut i = 0_usize;
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();

        while i + 16 <= n {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(v0)));
            acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(v0, 1)));
            acc2 = _mm256_add_pd(acc2, _mm256_cvtps_pd(_mm256_castps256_ps128(v1)));
            acc3 = _mm256_add_pd(acc3, _mm256_cvtps_pd(_mm256_extractf128_ps(v1, 1)));
            i += 16;
        }

        while i + 8 <= n {
            let value = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(value)));
            acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(value, 1)));
            i += 8;
        }

        let combined = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        let mut sum = hsum_pd(combined);
        while i < n {
            sum += f64::from(x[i]);
            i += 1;
        }
        sum
    }
}

/// Runtime-dispatched `avx2_fma` variant of [`sum_reduce_scalar`].
pub fn sum_reduce_avx2_fma(x: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            announce("sum_reduce", true);
            return unsafe { sum_reduce_avx2_impl(x) };
        }
    }
    announce("sum_reduce", false);
    sum_reduce_scalar(x)
}

/// Rayon data-parallel variant of [`sum_reduce_scalar`].
pub fn sum_reduce_rayon(x: &[f32]) -> f64 {
    use rayon::prelude::*;

    x.par_iter().map(|&value| f64::from(value)).sum()
}

// ---------------------------------------------------------------------------
// mandelbrot: iteration count of z <- z^2 + c over a width x height grid
// ---------------------------------------------------------------------------

/// Fixed escape radius squared; the loop continues while `|z|^2 <= 4`.
const MANDELBROT_ESCAPE: f32 = 4.0;
const X_MIN: f32 = -2.0;
const X_MAX: f32 = 1.0;
const Y_MIN: f32 = -1.5;
const Y_MAX: f32 = 1.5;

/// Counts completed iterations for a single complex point.
#[inline]
fn mandelbrot_pixel(cx: f32, cy: f32, max_iter: usize) -> usize {
    let mut zx = 0.0_f32;
    let mut zy = 0.0_f32;
    let mut k = 0_usize;
    while k < max_iter && zx * zx + zy * zy <= MANDELBROT_ESCAPE {
        let next_zx = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = next_zx;
        k += 1;
    }
    k
}

/// Sequential reference with opaque grid dimensions; per-pixel branch keeps
/// LLVM from vectorizing the data-dependent escape loop.
pub fn mandelbrot_scalar(width: usize, height: usize, max_iter: usize) -> f64 {
    let width = black_box(width);
    let height = black_box(height);
    let dx = (X_MAX - X_MIN) / width as f32;
    let dy = (Y_MAX - Y_MIN) / height as f32;
    let mut total = 0.0_f64;
    let mut row = 0_usize;
    while row < height {
        let cy = Y_MIN + row as f32 * dy;
        let mut col = 0_usize;
        while col < width {
            let cx = X_MIN + col as f32 * dx;
            total += mandelbrot_pixel(cx, cy, max_iter) as f64;
            col += 1;
        }
        row += 1;
    }
    total
}

/// Safe loop form. LLVM generally does not vectorize this kernel because the
/// inner escape loop is data-dependent and contains a data-dependent branch;
/// this is the honest `autovec` baseline for mandelbrot (see README).
pub fn mandelbrot_autovec(width: usize, height: usize, max_iter: usize) -> f64 {
    let dx = (X_MAX - X_MIN) / width as f32;
    let dy = (Y_MAX - Y_MIN) / height as f32;
    let mut total = 0.0_f64;
    for row in 0..height {
        let cy = Y_MIN + row as f32 * dy;
        for col in 0..width {
            let cx = X_MIN + col as f32 * dx;
            total += mandelbrot_pixel(cx, cy, max_iter) as f64;
        }
    }
    total
}

/// AVX2 + FMA implementation processing eight pixels at a time; only call
/// through [`mandelbrot_avx2_fma`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mandelbrot_avx2_impl(width: usize, height: usize, max_iter: usize) -> f64 {
    use core::arch::x86_64::*;

    unsafe {
        let dx = (X_MAX - X_MIN) / width as f32;
        let dy = (Y_MAX - Y_MIN) / height as f32;
        let escape = _mm256_set1_ps(MANDELBROT_ESCAPE);
        let two = _mm256_set1_ps(2.0);
        let one = _mm256_set1_ps(1.0);
        let mut total = 0.0_f64;

        for row in 0..height {
            let cy = Y_MIN + row as f32 * dy;
            let cy_v = _mm256_set1_ps(cy);
            let mut col = 0_usize;

            while col + 8 <= width {
                let base = col as f32;
                let cx = _mm256_setr_ps(
                    X_MIN + base * dx,
                    X_MIN + (base + 1.0) * dx,
                    X_MIN + (base + 2.0) * dx,
                    X_MIN + (base + 3.0) * dx,
                    X_MIN + (base + 4.0) * dx,
                    X_MIN + (base + 5.0) * dx,
                    X_MIN + (base + 6.0) * dx,
                    X_MIN + (base + 7.0) * dx,
                );

                let mut zx = _mm256_setzero_ps();
                let mut zy = _mm256_setzero_ps();
                let mut count = _mm256_setzero_ps();
                let mut k = 0_usize;

                while k < max_iter {
                    let zx2 = _mm256_mul_ps(zx, zx);
                    let zy2 = _mm256_mul_ps(zy, zy);
                    let mag = _mm256_add_ps(zx2, zy2);
                    let active = _mm256_cmp_ps(mag, escape, _CMP_LE_OS);
                    if _mm256_movemask_ps(active) == 0 {
                        break;
                    }
                    let zxy = _mm256_mul_ps(zx, zy);
                    zy = _mm256_fmadd_ps(two, zxy, cy_v);
                    zx = _mm256_add_ps(_mm256_sub_ps(zx2, zy2), cx);
                    count = _mm256_add_ps(count, _mm256_and_ps(active, one));
                    k += 1;
                }

                total += f64::from(hsum_ps(count));
                col += 8;
            }

            while col < width {
                let cx = X_MIN + col as f32 * dx;
                total += mandelbrot_pixel(cx, cy, max_iter) as f64;
                col += 1;
            }
        }

        total
    }
}

/// Runtime-dispatched `avx2_fma` variant of [`mandelbrot_scalar`].
pub fn mandelbrot_avx2_fma(width: usize, height: usize, max_iter: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && has_fma() {
            announce("mandelbrot", true);
            return unsafe { mandelbrot_avx2_impl(width, height, max_iter) };
        }
    }
    announce("mandelbrot", false);
    mandelbrot_scalar(width, height, max_iter)
}

/// Rayon data-parallel variant of [`mandelbrot_scalar`], one task per row.
pub fn mandelbrot_rayon(width: usize, height: usize, max_iter: usize) -> f64 {
    use rayon::prelude::*;

    let dx = (X_MAX - X_MIN) / width as f32;
    let dy = (Y_MAX - Y_MIN) / height as f32;

    (0..height)
        .into_par_iter()
        .map(|row| {
            let cy = Y_MIN + row as f32 * dy;
            let mut row_total = 0.0_f64;
            for col in 0..width {
                let cx = X_MIN + col as f32 * dx;
                row_total += mandelbrot_pixel(cx, cy, max_iter) as f64;
            }
            row_total
        })
        .sum()
}

// ---------------------------------------------------------------------------
// x86-64 horizontal reductions
// ---------------------------------------------------------------------------

/// Horizontal sum of the four `f64` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_pd(value: core::arch::x86_64::__m256d) -> f64 {
    use core::arch::x86_64::*;

    let low = _mm256_castpd256_pd128(value);
    let high = _mm256_extractf128_pd(value, 1);
    let pair = _mm_add_pd(low, high);
    let both = _mm_hadd_pd(pair, pair);
    _mm_cvtsd_f64(both)
}

/// Horizontal sum of the eight `f32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps(value: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;

    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let pair = _mm_add_ps(low, high);
    let quad = _mm_hadd_ps(pair, pair);
    let all = _mm_hadd_ps(quad, quad);
    _mm_cvtss_f32(all)
}
