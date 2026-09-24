# Benchmark output contract (v2.1)

Every benchmark in this lab **must** obey this contract so `bench/run_all.sh`
can aggregate results across C++, Rust, Go, and Python into one table.

- **v2** added checksum-timing, reproducibility, and cross-language rules.
- **v2.1** adds variant-classification, thread-count, merge-precedence, and
  dispersion rules from the Phase 1 re-review.

## Requirement: runnable + self-verifying

- Each benchmark runs a compute kernel over `n` elements and reports a checksum.
- The checksum is mandatory: it prevents the compiler/DCE from deleting the
  measured work, and it lets the harness confirm every variant computed the
  **same** result (correctness, not just speed).
- Checksum agreement is validated **within a language**, against that language's
  own `scalar` reference for the same `(example, n)`. (See *Cross-language*.)
- Benchmarks accept an optional size override: `--n <count>` (default per
  example), or env var `BENCH_N`. `n <= 0` must be rejected with a clear error.

## Required stdout line

One line per measured variant, comma-separated, exact field order:

```text
BENCH,<example>,<variant>,<n>,<ms>,<checksum>
```

| Field | Meaning |
|---|---|
| `example` | kernel name (see canonical kernels below) |
| `variant` | implementation name (see canonical list below) |
| `n` | number of elements processed |
| `ms` | **median** wall-clock milliseconds of the timed region (see rules below) |
| `checksum` | stable numeric digest of the output |

Stdout must contain `BENCH,` lines and **nothing else**. All human/log/machine
context goes to stderr.

## Canonical variant names (reuse these exact tokens)

`scalar`, `autovec`, `sse2`, `avx2`, `avx2_fma`, `avx512`, `openmp`, `pthreads`,
`std_threads`, `rayon`, `goroutines`, `pure_python`, `numpy`, `numba`,
`multiprocessing`, `ctypes_scalar`, `ctypes_simd`.

## Canonical kernels

`vector_add`, `saxpy`, `dot_product`, `sum_reduce`, `mandelbrot`,
`false_sharing_adjacent`, `false_sharing_padded`.

`memcpy_bw` is a **reserved** kernel token; it is **not implemented** in this
lab. The implemented coverage per language is the matrix in `docs/RESULTS.md`.

For the false-sharing demo, use the **example** names `false_sharing_adjacent`
and `false_sharing_padded` with the canonical variant `openmp` (do not invent new
variant tokens).

### `scalar` is a within-language reference, not a cross-language baseline

`scalar` means "this language's reference for tolerance comparison". Its codegen
varies: the C++ lane guarantees genuinely scalar code (vectorizers disabled via
`-fno-tree-vectorize`), whereas in Rust (and Go) a plain loop may still be
auto-vectorized by the backend. Do **not** read a cross-language `scalar` row as
a like-for-like non-vectorized baseline. For a true scalar/non-vectorized
comparison, use the C++ lane.

## Variant classes (single-thread vs multi-thread)

- **Single-thread variants**: `scalar`, `autovec`, `sse2`, `avx2`, `avx2_fma`,
  `avx512`, `pure_python`, `numpy`, `numba`, `ctypes_scalar`, `ctypes_simd`.
- **Multi-thread variants**: `openmp`, `pthreads`, `std_threads`, `rayon`,
  `goroutines`, `multiprocessing`.
- Any token not listed defaults to **single-thread**.

## Checksum semantics and timing

The checksum must be **independent of `REPS`** and describe a single application
of the kernel.

- **Element-wise kernels** (`vector_add`, `saxpy`; and `memcpy_bw` when
  implemented): the timed region
  is the element-wise write loop **only**. Compute the checksum in a separate
  **untimed** pass after timing. Do **not** perform an `O(n)` reduction inside
  the timed closure.
- **Reduction kernels** (`dot_product`, `sum_reduce`): the reduction *is* the
  kernel. Its scalar result is the checksum and is produced inside the timed
  region. Use the same accumulator strategy for every variant.
- **Output kernels** (`mandelbrot`, `false_sharing_*`): the checksum is the
  kernel output.
- **In-place kernels** (`saxpy`): must be `REPS`-invariant. Either use an
  out-of-place output buffer, or reset the destination from a pristine copy
  **before each timed rep, untimed**.

## Measurement rules (non-negotiable)

1. **Warm up** before timing (at least once untimed; reset in-place kernels
   untimed after warmup).
2. Report the **median** of `REPS` (default `REPS=7`) timed runs. Median of an
   even count is the **average of the two middle samples**.
3. Time only the kernel; allocate buffers beforehand.
4. Use a monotonic high-resolution clock (`std::chrono::steady_clock`,
   `Instant`, `time.perf_counter_ns`, Go `time.Now`).
5. Print machine/build context once to **stderr**: compiler version, SIMD flags,
   thread count. Never pollute the `BENCH,` stdout stream.
6. Checksums must defeat dead-code elimination (accumulate + observe /
   `black_box`).
7. **Report dispersion unconditionally** to stderr: at least `min` alongside the
   median (do not gate it behind `--verbose`). Comparison and tie decisions use
   the reported **median**; `min` is an advisory stability diagnostic.
8. Reject `n <= 0`.

## Pinning, thread count, and merging runs

On this hybrid CPU, exclude the E-cores (CPU 12-15) and state physical-vs-logical
threads explicitly. CPU 0-11 are **6 physical P-cores × 2 SMT threads** (e.g. CPU
2 and CPU 3 share one core).

- **Single-thread runs**: pin to one P-core, e.g. `taskset -c 2`, and leave its
  SMT sibling idle where practical.
- **Multi-thread runs**: pin to the P-core logical set `taskset -c 0-11` and set
  the thread count **explicitly**, identically across languages:
  `THREADS=6` (physical P-cores: use `taskset -c 0,2,4,6,8,10`) or `THREADS=12`
  (with SMT: `taskset -c 0-11`). Whichever is chosen, every language uses the
  same value and reports it. Set `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`,
  `GOMAXPROCS`, and the Python pool size to `THREADS`.
- Pinning a whole run to a single core collapses `openmp`/`rayon`/`goroutines`
  to one thread and **must not** be used for multi-thread numbers.
- Because one binary emits both classes, run it **twice** (once per class).

### Merge precedence (mandatory)

Single-thread variant rows are taken **only** from the single-thread run;
multi-thread variant rows are taken **only** from the multi-thread run. Never
mix. The aggregator keys rows by `(example, variant)` and applies this class
rule before comparing or rendering.

### Noise

With `powersave` + turbo, even pinned medians vary ~10-15%. Report `min` as an
advisory stability diagnostic; **flag cross-implementation differences under
~15% as ties** and never narrate an ordering inside the noise band as an effect.

## Cross-language comparison

Input generators differ between languages by design, so **checksums are not
expected to match across languages**. Therefore:

- the correctness gate is **within-language** (variant vs that language's own
  `scalar`, same `example` and `n`); and
- the cross-language table (`docs/RESULTS.md`) compares **time only**, at an
  explicit, identical `n` and identical `THREADS`.
- Kernels whose per-variant parameters differ across languages (e.g. Mandelbrot
  `MAX_ITER`/viewport) must either be aligned before comparison or be **excluded
  from cross-language time ranking** and annotated with their parameters.

## Numerical tolerance

FMA and reassociated/parallel reductions do **not** produce bit-identical results
to a scalar left-to-right loop. Compare checksums with a **true relative
tolerance of `1e-4`**, denominator `max(|reference|, 1e-12)`, for floating-point
kernels (exact equality for integers). Within tolerance is `ok`; outside is a
correctness failure. Never compare floats with `==`.

## Harness aggregation

`bench/run_all.sh` collects all `BENCH,` lines (ST and MT runs), enforces the
class-based merge precedence, checks each variant against its own language's
`scalar` within tolerance, keeps `results/results.csv`, and renders
`docs/RESULTS.md` (time-only across languages, ties flagged). A checksum outside
tolerance is a **correctness failure**, not a speed win.
