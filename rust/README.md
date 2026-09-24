# Rust lane — SIMD & parallel kernels (`simd-lab`)

Stable-Rust implementation of the lab's canonical kernels, emitting the frozen
`BENCH,` wire format from `docs/BENCH_CONTRACT.md` (v2). The explicit SIMD paths
use `core::arch::x86_64` intrinsics (AVX2 + FMA) behind a runtime feature check;
data parallelism uses [Rayon](https://crates.io/crates/rayon).

## Build & run

```bash
# from the repo root
cargo build --release --manifest-path rust/Cargo.toml          # plain build
RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --manifest-path rust/Cargo.toml        # AVX2 auto-vectorization

# one benchmark (BENCH lines on stdout; context on stderr)
cargo run --release --manifest-path rust/Cargo.toml --bin vector_add -- --verbose

# all five, in order
make -C rust run

# pin a single-thread comparison to a P-core (logical CPUs 0-11 on this box)
taskset -c 2 ./target/release/vector_add --verbose
```

`make run` writes **only** `BENCH,` lines to stdout; the `===== <bin> =====`
banners and all context go to stderr, so the stdout stream can be piped
straight into `bench/run_all.sh`.

Environment / flags:

| Control | Effect |
|---|---|
| `BENCH_N` or `--n <count>` | element count (per-example default below); `n <= 0` is rejected with exit status 2 |
| `BENCH_REPS` | timed repetitions, default `7`; median is reported |
| `--verbose` | human-readable summary (median, min, checksum) to **stderr** |

Per-example defaults: `vector_add`, `saxpy`, `dot_product`, `sum_reduce` use
`8_000_000` elements; `mandelbrot` uses `1024 × 768 = 786_432` pixels with
`MAX_ITER = 100`. For mandelbrot the width is fixed at 1024 and the height is
derived from the requested count, so the emitted `n` is the actual pixel count.

## What is timed, and when the checksum is computed

The harness runs one untimed warmup, then `BENCH_REPS` `Instant`-timed
repetitions, and reports the **median** (average of the two middle samples when
`REPS` is even). It follows the contract's checksum-timing split:

* **Element-wise kernels (`vector_add`, `saxpy`)**: the timed region is the
  element-wise write loop **only**. The checksum is an `O(n)` `f64` sum computed
  in a separate **untimed** pass after timing. Inside each timed rep the output
  buffer is passed through `black_box`, so the compiler must materialize the
  stores; the untimed checksum pass then reads the buffer so nothing is elided.
  No `O(n)` reduction is ever inside the timed closure.
* **Reduction kernels (`dot_product`, `sum_reduce`)**: the reduction *is* the
  kernel, so it stays inside the timed region and its `f64` result is the
  checksum.
* **Output kernel (`mandelbrot`)**: the kernel's iteration-count output is the
  checksum.

The machine/build context line and each variant's `median`, `min`, and checksum
are printed to **stderr unconditionally** (contract rules 5 and 7); `--verbose`
additionally prints the per-variant `ok`/`FAIL` agreement lines against `scalar`.

## Variants

| Token | Meaning |
|---|---|
| `scalar` | The straightforward indexed loop; the reference for checksum comparison. It is **not** a promise of scalar machine code — see the vectorization section below. |
| `autovec` | Safe iterator code written so LLVM *can* auto-vectorize it. |
| `avx2_fma` | Explicit 256-bit intrinsics under `#[target_feature(enable = "avx2,fma")]`, reached through a runtime check. |
| `rayon` | The same math split across Rayon's work-stealing thread pool. |

### The AVX2/FMA variant is `unsafe`

Every intrinsic implementation is declared exactly as:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vector_add_avx2_impl(a: &[f32], b: &[f32], out: &mut [f32]) { /* ... */ }
```

The safety contract has two halves:

1. **`unsafe fn`** — the function may only be called if the running CPU actually
   supports AVX2 and FMA. Executing those instructions on a CPU without them is
   undefined behaviour, which is why the compiler requires an `unsafe` block.
2. **Call only after runtime detection** — the public wrapper checks
   `is_x86_feature_detected!("avx2")` *and* `is_x86_feature_detected!("fma")`
   before the `unsafe` call. `is_x86_feature_detected!` is cached in an atomic
   by `std`, so the check is cheap. On a CPU without AVX2, or on a non-x86-64
   target, the wrapper runs the `scalar` reference instead and says so on
   stderr:

   ```text
   [rust] vector_add: avx2_fma -> AVX2+FMA detected, using core::arch intrinsics
   [rust] vector_add: avx2_fma -> AVX2/FMA unavailable, falling back to scalar
   ```

`_mm256_loadu_ps` / `_mm256_storeu_ps` are used because inputs have no
guaranteed 32-byte alignment; each kernel handles the `n % 8` tail with a scalar
remainder loop.

## Why not `std::simd`?

`std::simd` (portable SIMD) is **nightly-only**; this crate targets stable
`1.96.1`, so it is not available. Instead the AVX2 paths use
`core::arch::x86_64`, which is stable and maps one-to-one onto the machine's
instructions. The trade-off is x86-64-specific code behind `#[cfg]`, which is
exactly why the runtime feature check and scalar fallback exist.

## How `RUSTFLAGS` changes the generated code

* **Without** `-C target-cpu=native`, the baseline target is generic `x86-64`
  (SSE2 only). LLVM auto-vectorizes `autovec` to 128-bit SSE2 at best, and
  `f32::mul_add` / `f64` reductions cannot use FMA. The explicit `avx2_fma`
  path still works, because `#[target_feature]` enables the instructions for
  that function regardless of the global target.
* **With** `RUSTFLAGS="-C target-cpu=native"`, the compiler may use AVX2 (256-bit
  YMM registers) and FMA for `autovec` and for `mul_add`. This is what the
  `make` targets set.

Check the effect on a kernel with:

```bash
objdump -d --no-show-raw-insn target/release/dot_product | grep -c ymm
```

## What the compiler actually did, and why

These are findings from inspecting generated assembly, not assumptions.

**`black_box(len)` does not stop the vectorizer.** It only hides the trip count
from constant folding and full unrolling. The real barrier to vectorizing the
reference loops is the **non-reassociable strict-`f64` reduction**. Rust does
not enable fast-math, so LLVM may not reorder a floating-point sum across
vector lanes. A controlled compile of the two reference loop shapes shows this
directly:

| Loop shape (with `black_box(len)`) | YMM instructions | Inner arithmetic |
|---|---|---|
| element-wise `out[i] = a[i] + b[i]` | **12** | `vaddps` (vectorized) |
| reduction `s += f64::from(a[i]) * f64::from(b[i])` | **0** | `vmulsd` (scalar) |

Consequences for the variants:

* **`scalar` vs `autovec` for element-wise kernels.** With the checksum moved
  out of the timed loop, `vector_add_scalar` / `saxpy_scalar` have no reduction,
  so LLVM vectorizes them too. `scalar ≈ autovec` for `vector_add`/`saxpy` is
  therefore expected; those kernels are memory-bandwidth bound, and the
  meaningful SIMD question is how they compare to `avx2_fma` and `rayon`.
* **Reductions stay scalar.** `dot_product_scalar` / `sum_reduce_scalar` keep
  the `f64` accumulation, and LLVM will not reassociate it, so they contain no
  YMM instructions. `dot_product_autovec` / `sum_reduce_autovec` use
  `.map(..).sum()` and are *also* not vectorized, for the same reason. Their
  timings are close to `scalar`; that is a real result, not a bug. It is exactly
  why the `avx2_fma` variant exists: it uses four independent `f64` accumulators,
  a reassociation the programmer chose explicitly.
* **`mandelbrot` does not auto-vectorize.** Its inner escape loop has a
  data-dependent trip count and a data-dependent branch, so `autovec` matches
  `scalar`; the explicit AVX2 path evaluates eight pixels per iteration with a
  lane mask and is much faster.
* **`rayon` is the other legal way to reassociate**: each thread reduces its own
  chunk in `f64`, then the partials are summed. That is the parallel analogue of
  the four-accumulator SIMD trick.

Sample run on the reference machine (`RUSTFLAGS="-C target-cpu=native"`,
`BENCH_REPS=7`, unpinned so Rayon can use all cores; your numbers will differ):

```text
BENCH,vector_add,scalar,8000000,4.0881,16499998.437500
BENCH,vector_add,autovec,8000000,3.4469,16499998.437500
BENCH,vector_add,avx2_fma,8000000,3.5134,16499998.437500
BENCH,vector_add,rayon,8000000,2.7987,16499998.437500
BENCH,dot_product,scalar,8000000,4.8406,8124998.273438
BENCH,dot_product,avx2_fma,8000000,2.0089,8124998.273438
BENCH,dot_product,rayon,8000000,1.2794,8124998.273438
BENCH,sum_reduce,avx2_fma,8000000,1.1409,9999999.187500
BENCH,mandelbrot,avx2_fma,786432,4.7822,16516739.000000
```

Pinning and noise: single-thread comparisons must run pinned to a P-core
(`taskset -c 2`); multi-thread variants must run pinned to the P-core set with an
explicit thread count (`RAYON_NUM_THREADS`) — pinning a whole run to one core
drops `available_parallelism()` to 1 and collapses Rayon to single-threaded, so
its numbers are only meaningful when the process may use multiple cores. The
measurement host runs the `powersave` governor with turbo enabled and medians
vary substantially run-to-run (unpinned `mandelbrot,scalar` medians ranged
35.5–68.6 ms across runs). The harness therefore **always** prints `min` to
stderr as the stability reference and flags differences under ~15% as ties. Do
not interpret orderings that sit inside that noise as effects.

## Correctness & tolerance

Checksums are compared **within this language only**, against the Rust `scalar`
reference for the same `(example, n)`. Input generators differ between
languages by design, so **checksums are not expected to match across languages**;
the cross-language table compares **time only**.

`avx2_fma` and `rayon` may reorder additions (FMA rounds once; Rayon reduces in
a different order), so results need not be bit-identical to `scalar`.
`--verbose` uses the contract's true relative tolerance of `1e-4` with
denominator `max(|reference|, 1e-12)` and prints `ok`/`FAIL` to stderr. A
mismatch outside tolerance is a correctness failure, never a speed win.

### `REPS`-invariance

The contract requires a checksum that does not change with `BENCH_REPS`. All
Rust kernels satisfy this:

* `vector_add`, `saxpy` are **out-of-place** (`out` is distinct from the
  inputs), so every timed rep reads pristine inputs and overwrites `out`; the
  untimed checksum pass then hashes a deterministic buffer.
* `dot_product`, `sum_reduce` are pure functions of their immutable inputs, and
  the checksum is the reduction result of a single application.
* `mandelbrot` is a pure function of the grid dimensions.

Verified: `BENCH_REPS=1` and `BENCH_REPS=7` produce identical checksums for
every kernel (e.g. `vector_add` and `saxpy` below).

```text
# BENCH_REPS=1
BENCH,vector_add,scalar,8000000,3.6621,16499998.437500
BENCH,saxpy,scalar,8000000,4.4965,31499997.218750
# BENCH_REPS=7
BENCH,vector_add,scalar,8000000,4.2940,16499998.437500
BENCH,saxpy,scalar,8000000,4.4037,31499997.218750
```

## Layout

```text
rust/
├── Cargo.toml          # package simd-lab, rayon 1.12, release profile
├── rust-toolchain.toml # pinned 1.96.1 + rustfmt + clippy
├── build.rs            # captures rustc version + RUSTFLAGS for stderr context
├── Makefile            # make build / run / run-verbose / clippy / fmt
└── src/
    ├── lib.rs          # harness + all kernels and dispatch wrappers
    └── bin/
        ├── vector_add.rs
        ├── saxpy.rs
        ├── dot_product.rs
        ├── sum_reduce.rs
        └── mandelbrot.rs
```
