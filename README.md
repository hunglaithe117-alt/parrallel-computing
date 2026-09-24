# Parallel Computing & SIMD Lab

A hands-on lab for learning **parallel computing** and **SIMD** across four
languages — **C++**, **Rust**, **Go**, and **Python** — with benchmarks that
measure speed on *your* machine and explanations of *why* the fast versions are
fast.

> Status: complete. All four language lanes, the cross-language harness, the
> conceptual chapters, and the measured results are in place. **Start with
> [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md)**, then read the measured
> [results](docs/RESULTS.md).

## Where to start

| Document | What it covers |
|---|---|
| [`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md) | Suggested study order C++ → Rust → Go → Python → cross-language, with exact commands |
| [`docs/CONCEPTS.md`](docs/CONCEPTS.md) | SIMD, AVX2/FMA, P/E cores, caches, Roofline |
| [`docs/WHY_FASTER.md`](docs/WHY_FASTER.md) | The sources of speedup, tied to the lab's kernels |
| [`docs/CPU_METHODS.md`](docs/CPU_METHODS.md) | Practical methods to make CPU code faster |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | How the benchmarks are measured (and their pitfalls) |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Generated: measured cross-language + within-language results |
| [`docs/BENCH_CONTRACT.md`](docs/BENCH_CONTRACT.md) | The `BENCH,` output contract every benchmark obeys |

## Why this machine matters

This lab is written for a specific CPU, because SIMD results are
hardware-specific. Check yours with `lscpu`.

| Property | This machine |
|---|---|
| CPU | Intel Core i5-14400 (Raptor Lake, hybrid P+E cores) |
| Threads | 16 (6 P-cores + 4 E-cores) |
| SIMD | SSE2 … **AVX2 + FMA + AVX-VNNI** (256-bit max, **no AVX-512**) |
| Compilers | gcc/g++ 15.2, clang 23.1, rustc 1.96 (stable), go 1.27.1 |
| Python | CPython 3.14 |

Because there is no AVX-512, all examples target 128-bit (SSE2) and 256-bit
(AVX2/FMA) vectors. Rust examples use stable `core::arch` intrinsics, not the
nightly-only `std::simd`.

## Layout

```text
parallel-computing-lab/
├── cpp/      # scalar vs auto-vectorized vs SSE2/AVX2/FMA intrinsics, OpenMP
├── rust/     # core::arch intrinsics, target_feature, Rayon
├── go/       # goroutines, GOMAXPROCS, cache behaviour, SIMD limitations
├── python/   # pure Python vs numpy vs multiprocessing vs ctypes -> C SIMD
├── bench/    # run_all.sh + aggregate.py: aggregate every language
├── docs/     # concepts, why-faster, CPU methods, methodology, results
└── results/  # generated: results.csv + raw captures + build logs
```

## Benchmark contract

Every benchmark emits `BENCH,<example>,<variant>,<n>,<ms>,<checksum>`.
Correctness is verified **within each language** against that language's own
`scalar` reference; the cross-language table compares **time only**, because the
input generators differ by design. See
[`docs/BENCH_CONTRACT.md`](docs/BENCH_CONTRACT.md).

## Quick start

```bash
# build + run one language (from the lab root)
make -C cpp && make -C cpp run          # C++
make -C rust run                        # Rust
make -C go run                          # Go
make -C python setup && make -C python lib && make -C python run   # Python

# run every language pinned, then regenerate docs/RESULTS.md
./bench/run_all.sh
```

Per-language details, variants, and pinning/thread conventions are in each
folder's `README.md`.
