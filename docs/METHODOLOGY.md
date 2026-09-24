# Methodology: how this lab measures

The benchmark contract (`BENCH_CONTRACT.md`) is the normative rulebook. This
chapter explains *why* each rule exists so you can apply the same discipline to
your own measurements.

## What is measured

Every benchmark runs one compute kernel over `n` elements, one untimed warm-up
pass, then `REPS` timed passes (default **7**). The reported `ms` is the
**median** wall-clock time of the timed region; for an even `REPS` the median is
the average of the two middle samples. Only the kernel is inside the clock:
buffers are allocated and filled beforehand, and any `O(n)` checksum sum is
computed afterwards, off the clock. The clock itself is monotonic and
high-resolution (`steady_clock`, Rust `Instant`, `time.perf_counter_ns`, Go
`time.Now`).

**Warm-up** exists because the first pass pays page faults, cold caches, and
lazily-resolved runtime paths. It is discarded. For the in-place `saxpy`
kernel, the destination buffer is also reset from a pristine copy before each
timed rep — untimed — so repeated reps do not change what is being measured.

**Median versus mean.** The median is robust to a single slow rep (another
process waking up, a frequency dip). The mean would let that rep dominate.

**`min` is an advisory stability diagnostic.** The contract requires every
program to print `min` alongside the median, unconditionally, but comparison and
tie decisions use the reported **median**. Median answers "what is typical"; min
answers "how fast this got when nothing interfered", which tells you how stable
the run was, not which variant won.

## Defeating dead-code elimination

The compiler is allowed to delete any computation whose result is never
observed. A benchmark that only calls the kernel would measure nothing. Two
guards are mandatory:

- The **checksum** is a numeric digest of the output. It forces the kernel's
  result to be computed and, because it is printed, observed.
- The accumulator is passed through an opaque barrier — `black_box` in C++/Rust,
  an exported sink variable in Go/Python — so the optimizer cannot prove the
  value is unused.

The checksum must also be **`REPS`-invariant**: it describes a single
application of the kernel, not an accumulation over repetitions. Element-wise
kernels compute it in a separate untimed pass; reduction kernels return the
single reduction result from inside the timed region (there, the reduction *is*
the kernel). An in-place kernel like `saxpy` resets its output each rep so its
digest does not scale with `REPS`.

## Pinning and thread count

On this hybrid CPU, results are only meaningful with explicit affinity. Contract
rules:

- **Single-thread variants**: `taskset -c 2` (one P-core).
- **Multi-thread variants**: `taskset -c 0-11` (the P-core logical set) with the
  thread count set **explicitly and identically** across languages:
  `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `GOMAXPROCS`, and the Python pool size
  all equal `THREADS`. `THREADS=12` uses SMT; `THREADS=6` with
  `taskset -c 0,2,4,6,8,10` uses one thread per physical P-core.

Why this is non-negotiable: a thread pool sizes itself from the process's
affinity mask. Pinning an *entire* run to a single core makes OpenMP / Rayon /
goroutines / `multiprocessing` see one CPU and collapse to one worker. Their
timings then equal the serial variant and are meaningless. The contract's fix is
to run each binary **twice** — once per variant class — and take single-thread
rows only from the single-thread run and multi-thread rows only from the
multi-thread run (**merge precedence**). Never mix the two.

E-cores (CPU 12–15) are excluded from both sets because they are slower and
128-bit; including them would make results depend on where the scheduler placed
a thread.

## Host noise

The benchmark host runs the `powersave` governor with turbo enabled
(`no_turbo=0`). Core frequency therefore drifts with load and thermal state, and
even pinned medians vary by roughly **10–15%** run to run. The rules that follow
from that:

- Report `min` as an advisory stability diagnostic; ranking and tie decisions
  use the median.
- **Flag any cross-implementation difference under ~15% as a tie.** Do not
  narrate an ordering inside the noise band as an effect. A 3% "win" is not a
  win.
- Re-run before believing a small effect; if it does not reproduce, it was
  noise.

If you need lower variance, the honest options are to change the governor
(`performance`) and disable turbo, or to raise `REPS` — not to
cherry-pick a favourable median.

## Correctness gate and tolerance

Speed means nothing if the variant computes something different. Correctness is
checked **within a language**, by comparing each variant's checksum against that
language's own `scalar` reference for the same `(example, n)`.

- Floating-point checksums use a **true relative tolerance of `1e-4`**, with
  denominator `max(|reference|, 1e-12)`.
- Integer kernels are compared exactly.
- Outside tolerance is a **correctness failure**, not a speed win.

The tolerance is required because FMA and reassociated/parallel reductions do
**not** produce bit-identical results to a scalar left-to-right loop:

- FMA rounds `a*b+c` once instead of twice (Intel SDM).
- Four-accumulator SIMD and thread-partial reductions sum in a different order.
- `saxpy` uses a deliberately non-dyadic `alpha`, so its FMA checksum differs
  from scalar in the last digits (C++: 7129389.843 scalar vs 7129389.835
  `avx2_fma` — well inside 1e-4).

Never compare floats with `==`.

## Cross-language comparison

Input generators differ between languages by design, so **checksums are not
expected to match across languages**. The cross-language table therefore
compares **time only**, at an explicit, identical `n` and identical `THREADS`.

Two traps specific to this lab:

- **Default `n` differs per language.** C++ defaults to 4.19 M / 16.8 M, Rust
  to 8 M, Go to 4.19 M / 16.8 M, and Python to 32 M / 56 M (chosen so its
  interpreted reference lands in the ~1–2 s band). A comparison must pass an
  explicit `n`; comparing defaults would compare different amounts of work.
- **Mandelbrot parameters differ per language.** `MAX_ITER` and the viewport
  are not aligned across lanes, so mandelbrot is either aligned before
  comparison or **excluded from cross-language time ranking** and annotated with
  its parameters. A pixel iteration count is not comparable across different
  iteration caps.

Also remember the contract's `scalar` rule: C++ `scalar` is genuinely
non-vectorized, while Rust/Go `scalar` may be auto-vectorized. `scalar` is a
within-language reference, never a cross-language baseline.

## Using `perf stat`

Timings tell you *that* a variant is faster; `perf stat` tells you *why*. Useful
counters: `cycles`, `instructions`, `branches`, `branch-misses`,
`cache-references`, `cache-misses`. IPC is `instructions / cycles`.

```bash
# single-thread, on one P-core
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  taskset -c 2 ./cpp/bin/dot_product

# multi-thread OpenMP run
perf stat -e cycles,instructions,cache-misses \
  env OMP_NUM_THREADS=12 taskset -c 0-11 ./cpp/bin/mandelbrot
```

This CPU is **hybrid**, so `perf` exposes separate PMUs for the two core types:
`cpu_core` (P-cores) and `cpu_atom` (E-cores). To attribute events per type, name
the PMU explicitly, e.g. `-e cpu_core/cycles/,cpu_atom/cycles/`; the bare event
name aggregates both. Check `perf list` for what this kernel exposes, and note
that event access may require a permissive `perf_event_paranoid` setting.

How to read it:

- High `cache-misses` relative to `cache-references` and IPC around or below 1
  is the signature of a memory/latency-bound kernel (large-N `vector_add`,
  `sum_reduce`).
- High IPC with few misses is compute-bound (`mandelbrot` with its small output
  array).
- A surprising cycle count with an expected instruction count points at stalls —
  false sharing, alignment faults, or AVX/SSE transition penalties (the Go
  experimental `simd` build on this machine is a documented example of the
  last one).

## Reproducibility checklist

1. Build clean; record compiler, version, and ISA flags to stderr.
2. Warm up once; time only the kernel; allocate before the clock.
3. `REPS=7`, take the median, print `min` unconditionally (advisory).
4. Pin per variant class; set the thread count explicitly.
5. Emit a `REPS`-invariant checksum; gate on within-language tolerance.
6. Compare across languages at identical `n` and `THREADS`, time only.
7. Treat differences under ~15% as ties, and re-run before believing them.
