# Go lane — goroutines, GOMAXPROCS & SIMD limitations

Go implementations of the lab kernels that emit the frozen
`BENCH,<example>,<variant>,<n>,<ms>,<checksum>` contract line
(see `../docs/BENCH_CONTRACT.md` v2.1). The Go lane's teaching point is the
opposite of the C++ lane's: instead of hand-written intrinsics, you get
**goroutines for parallelism and nothing for SIMD**, plus a look at what the
experimental `simd` package does and does not buy you on Go 1.27.

Target machine (results only mean anything on this class of CPU):

* Intel Core i5-14400 (Raptor Lake hybrid): P-cores CPU 0–11 (6 physical × 2
  SMT), E-cores CPU 12–15. The contract runs exclude the E-cores.
* SIMD: SSE2, **AVX2 (256-bit) + FMA3**, no AVX-512.
* go 1.27.1, gc compiler, `GOAMD64=v3` baseline (AVX2 + FMA).

## Build and run

```bash
make -C go            # build contract binaries in bin/, tool in bin-tools/
make -C go run        # run every benchmark, print BENCH lines
make -C go scale      # GOMAXPROCS 1..12 scaling table for the reductions
make -C go simd       # experimental GOEXPERIMENT=simd build (see below)
make -C go clean
```

Equivalently, without make:

```bash
GOAMD64=v3 go build ./...
GOAMD64=v3 go build -o bin/vector_add ./cmd/vector_add
```

Per-example options (`--n` overrides `BENCH_N` overrides the compiled default;
`n <= 0` is rejected; `BENCH_REPS` selects the timed-run count):

```bash
taskset -c 2 ./bin/vector_add --n 1048576
BENCH_N=1048576 BENCH_REPS=11 taskset -c 2 ./bin/dot_product
GOMAXPROCS=12 taskset -c 0-11 ./bin/sum_reduce
taskset -c 0-11 ./bin-tools/scale
```

### Pinning and thread count (contract)

Go reports `runtime.NumCPU()` through the process CPU affinity mask, so
`taskset -c 2` makes Go see one CPU and default `GOMAXPROCS` to 1. The worker
pools therefore collapse to a single worker on a single-thread run, exactly as
the merge-precedence rule expects; only the multi-thread run's `goroutines` rows
are used. The contract multi-thread convention is `THREADS=12`:

```bash
# single-thread: one P-core (CPU 2, sibling CPU 3 idle)
taskset -c 2 ./bin/vector_add

# multi-thread: all 6 P-cores / 12 SMT threads, count set explicitly
GOMAXPROCS=12 taskset -c 0-11 ./bin/vector_add
```

`GOMAXPROCS` is the thread count Go schedules on; the kernels size their worker
pool from it (`bench.Threads()`), and the harness prints it to stderr on every
run.

## Why Go does not auto-vectorize

The gc compiler deliberately has no auto-vectorizer
(<https://go.dev/wiki/CompilerOptimizations>): a plain `for` loop over a slice
compiles to scalar instructions (on this machine, `MOVSS`/`CVTSS2SD` plus a
scalar `VFMADD...SD`), no matter the `GOAMD64` level. That is why this lane has
no `autovec` variant: the `scalar` functions are both the language reference and
the honest non-SIMD baseline. Any 256-bit speedup has to come from an explicit
vector API — or from threads.

`GOAMD64=v3` still matters: it makes AVX2 + FMA part of the *baseline*
instruction set, so the archsimd path and the runtime may use those instructions
without per-call feature dispatch. It does **not** make the compiler
auto-vectorize.

## Variants

| Example | Variants | What it demonstrates |
|---|---|---|
| `vector_add` | `scalar`, `goroutines` (+ `avx2_fma`) | memory-bound element-wise work; goroutine split across disjoint output ranges |
| `dot_product` | `scalar`, `goroutines` (+ `avx2_fma`) | serial FMA chain vs one partial sum per worker; four vector FMA accumulators |
| `sum_reduce` | `scalar`, `goroutines` (+ `avx2_fma`) | same story on a single array |
| `false_sharing_adjacent` | `goroutines` | packed `[]int64` counters share cache lines → ping-pong |
| `false_sharing_padded` | `goroutines` | one 64-byte cache line per counter → no ping-pong |

`scalar` is the within-language checksum reference; the harness compares every
variant against it with the contract's relative tolerance of `1e-4`.

### Goroutine worker pools

The `goroutines` kernels split `[0, n)` into contiguous chunks and run one
goroutine per chunk, synchronised with `sync.WaitGroup`. Each worker writes only
its own output range; nothing is appended to a shared slice. The reductions give
each worker a slot in a **preallocated** `partials []float64` scratch buffer
(allocated before timing, never inside the timed region) and sum the slots after
the wait, so every variant uses a float64 accumulator and the parallel result
stays inside tolerance. Per-worker float64 partials reduce the reassociation
error compared with a single shared accumulator.

### False sharing and 64-byte padding

Go has no `volatile`, so the counters use `atomic.AddInt64`: a real
read-modify-write to memory every iteration, which is what makes the cache-line
traffic visible. `false_sharing_adjacent` stores `[]int64`, so eight counters
share a 64-byte line and the workers' atomic increments invalidate each other's
line. `PaddedCounter` is `int64` + `[56]byte` = exactly 64 bytes; a compile-time
assertion fails the build if that ever changes. Same total both ways, very
different time (roughly 10× in the sample below).

## The experimental `simd` package

Go 1.27 ships `simd` / `simd/archsimd` only behind `GOEXPERIMENT=simd`. It is
explicitly **unstable and not covered by the Go 1 compatibility promise**; GA is
deferred to Go 1.28. `Float32x8` lowers to 256-bit AVX2 registers and `MulAdd`
lowers to `VFMADD...PS`.

The `avx2_fma` functions live in `internal/kernels/simd_enabled.go` behind
`//go:build goexperiment.simd`; `simd_disabled.go` (build tag
`!goexperiment.simd`) provides the same symbols and sets `AVX2FMAEnabled =
false`. The default `go build ./...` therefore **never** needs the experiment,
and the example binaries omit the `avx2_fma` rows unless they were built with it:

```bash
GOEXPERIMENT=simd GOAMD64=v3 go build ./...
GOEXPERIMENT=simd GOAMD64=v3 go build -o bin-simd/vector_add ./cmd/vector_add
GOEXPERIMENT=simd GOAMD64=v3 ./bin-simd/vector_add
```

`make -C go simd` compiles everything with the experiment and writes the
binaries to `bin-simd/` (never over `bin/`), so a default contract run cannot
accidentally pick up an experimental binary.

### Known Go 1.27.1 artifact: AVX→SSE transition penalties

The experiment build works and every `avx2_fma` checksum matches `scalar` within
tolerance, but it has a serious performance artifact that is itself instructive:
under `GOEXPERIMENT=simd` the compiler emits some 256-bit instructions
**somewhere in runtime/startup** that leave the upper YMM bits dirty, and this
machine then penalises the *legacy SSE* instructions in the scalar loops
(`MOVSS`/`CVTSS2SD` are not VEX-encoded). The effect is not uniform —
`vector_add`/`sum_reduce` scalar stay normal, while `dot_product` scalar slows by
~100× (≈400 ms vs ≈4 ms). It happens even without importing `archsimd`, and
`archsimd.ClearAVXUpperBits()` only restores speed on a second call, too fragile
to depend on.

Consequences:

* Treat the experimental build's `avx2_fma` rows and all checksums as valid, but
  **take scalar timings (and cross-language comparisons) from the default
  build**; the simd build's scalar/legacy-SSE rows are not a fair baseline.
* This is one more reason the experiment is GA-deferred. The default build —
  the one the contract harness uses — is unaffected.

## Contract compliance notes

* **Timing**: `time.Now`/`time.Since`, kernel only; buffers allocated and filled
  before `Measure`. One untimed warm-up, then `BENCH_REPS` (default 7) timed
  runs; `ms` is the median (average of the two middle samples for an even
  count). `min` and the build/thread context are always printed to stderr,
  never gated behind `--verbose`.
* **Checksums and `REPS`-invariance**: element-wise `vector_add` writes its whole
  output buffer from `a`/`b` each call and sums it in a **separate untimed
  pass**, so the digest does not depend on how many times it ran. The reductions
  return a deterministic float64 scalar from inside the timed region; the
  false-sharing total is `iters × workers` with counters reset at the start of
  every invocation. Empirically, `BENCH_REPS=1` and `BENCH_REPS=7` produce
  byte-identical checksums (shown in the lane results).
* **Sink**: every measured result is accumulated into the exported
  `bench.Sink`, so the compiler cannot delete the work as dead code.
* **`n` precedence**: `--n`, then `BENCH_N`, then the per-example default;
  `n <= 0` (and non-numeric input) exits with status 2.
* **Variant classes**: `scalar` is single-thread; `goroutines` is multi-thread;
  `avx2_fma` is single-thread. Run the binary once pinned to one P-core and once
  pinned to `0-11` with `GOMAXPROCS=12`, and let the aggregator apply merge
  precedence.

## Sample results (this machine)

Single-thread (`taskset -c 2`), default build:

```text
BENCH,vector_add,scalar,4194304,2.0872,8387579.032
BENCH,vector_add,goroutines,4194304,2.4168,8387579.032
BENCH,dot_product,scalar,4194304,4.2511,4194553.685
BENCH,dot_product,goroutines,4194304,4.3849,4194553.685
BENCH,sum_reduce,scalar,16777216,8.7386,16778875.29
BENCH,sum_reduce,goroutines,16777216,8.7209,16778875.29
BENCH,false_sharing_adjacent,goroutines,2000000,8.9160,2000000
BENCH,false_sharing_padded,goroutines,2000000,8.9494,2000000
```

Multi-thread (`GOMAXPROCS=12 taskset -c 0-11`), default build:

```text
BENCH,vector_add,scalar,4194304,2.2851,8387579.032
BENCH,vector_add,goroutines,4194304,1.2550,8387579.032
BENCH,dot_product,scalar,4194304,4.1666,4194553.685
BENCH,dot_product,goroutines,4194304,0.5795,4194553.685
BENCH,sum_reduce,scalar,16777216,8.6811,16778875.29
BENCH,sum_reduce,goroutines,16777216,1.9118,16778875.29
BENCH,false_sharing_adjacent,goroutines,24000000,212.5227,24000000
BENCH,false_sharing_padded,goroutines,24000000,21.4190,24000000
```

The reductions scale to about 5–5.5× on 12 SMT threads because they are
memory-bandwidth-bound, not FLOP-bound; the false-sharing pair makes the
cache-line effect obvious. The host runs `powersave` with turbo, so medians vary
~10–15% run to run — `min` is the stability reference and differences inside
that band are ties.

## Layout

```text
go/
├── go.mod                       module simdlab, go 1.27
├── Makefile                     build / run / scale / simd / clean
├── internal/
│   ├── bench/bench.go           contract harness (timing, median, emit, sink)
│   └── kernels/
│       ├── kernels.go           scalar + goroutines kernels, padded counter
│       ├── simd_enabled.go      //go:build goexperiment.simd
│       └── simd_disabled.go     //go:build !goexperiment.simd
├── bin/                         contract example binaries only (BENCH stdout)
├── bin-tools/scale              non-BENCH tool, kept out of bin/ globs
├── bin-simd/                    experimental build output (make simd)
└── cmd/
    ├── vector_add/main.go
    ├── dot_product/main.go
    ├── sum_reduce/main.go
    ├── false_sharing_adjacent/main.go
    ├── false_sharing_padded/main.go
    └── scale/main.go            GOMAXPROCS scaling table (tool, not contract)
```
