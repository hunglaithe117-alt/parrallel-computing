# C++ lane — SIMD & OpenMP learning lab

Hand-written scalar, auto-vectorized, SSE2/AVX2/FMA and OpenMP kernels that all
emit the frozen `BENCH,` contract line (see `../docs/BENCH_CONTRACT.md`). The
point is to *see* what each level of parallelism buys and why.

Target machine (the results only mean anything on this class of CPU):

* Intel Core i5-14400 (Raptor Lake hybrid: 12 P-threads + 4 E-threads)
* SIMD: SSE2, **AVX2 (256-bit) + FMA3**, no AVX-512
* g++ 15.2, `-march=native` expands to AVX2 + FMA

## Build and run

```bash
make -C cpp            # build bin/<kernel> for every kernel
make -C cpp run        # run all kernels, print every BENCH line to stdout
make -C cpp vec-report # show what GCC auto-vectorized and what it missed
make -C cpp disasm     # objdump proof of vaddps / vfmadd in the AVX2 kernels
make -C cpp clean
```

Per-kernel options (build context always goes to stderr, never stdout):

```bash
./bin/vector_add --n 1048576 --verbose
BENCH_N=1048576 BENCH_REPS=11 ./bin/dot_product
```

`--n <count>` overrides `BENCH_N` overrides the compiled default; `n <= 0` is
rejected. `BENCH_REPS` selects the number of timed runs (default 7); the
reported `ms` is the **median** (average of the two middle samples for even
`REPS`), after one untimed warm-up. Time covers the compute region only:
buffers are allocated and filled beforehand, and the checksum is summed after
the clock stops. For the in-place `saxpy`, the destination is restored from a
pristine copy before each timed rep (untimed), so its checksum is a single
application and is invariant to `BENCH_REPS`.

## Variants

| Kernel | Variants | What it demonstrates |
|---|---|---|
| `vector_add` | `scalar`, `autovec`, `sse2`, `avx2_fma` | memory-bound element-wise work; 128- vs 256-bit |
| `saxpy` | `scalar`, `autovec`, `avx2_fma` | FMA contraction (`vfmadd`), non-dyadic alpha shows the rounding change |
| `dot_product` | `scalar`, `autovec`, `avx2_fma` | serial reduction vs four independent FMA accumulators (ILP) |
| `sum_reduce` | `scalar`, `autovec`, `avx2_fma` | same story with a single array; 16 Mi elements |
| `mandelbrot` | `scalar`, `avx2_fma`, `openmp` | compute-bound + branch divergence; 8 pixels/iteration vs rows across threads |
| `false_sharing_adjacent` / `false_sharing_padded` | `openmp` each | cache-line ping-pong: same result, very different time |

`scalar` is the checksum reference for every kernel. The harness (and each
program's own stderr check) compares checksums with the contract's relative
tolerance of **1e-4**.

## How the variants are compiled

Compiler flags are per translation unit, so each variant lives in its own
`.cpp` and `kernels.h` declares the shared interface. The Makefile links the
groups into one binary per kernel:

| Translation unit | Flags |
|---|---|
| `scalar_kernels.cpp` | `-O3 -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off` |
| `autovec_kernels.cpp` | `-O3 -march=native` |
| `simd_kernels.cpp` | `-O3 -mavx2 -mfma` |
| `openmp_kernels.cpp` | `-O3 -march=native -fopenmp` |

No `-ffast-math` in the default build, so floating-point results stay
comparable with the contract tolerance.

## Inspecting the generated code

Ask GCC what it did with the plain loops:

```bash
g++ -O3 -std=c++17 -march=native -fopt-info-vec-missed -c src/autovec_kernels.cpp -o /dev/null
# or: make vec-report   (shows optimized loops and the missed ones)
```

Look for the actual AVX2/FMA instructions in the intrinsics kernels:

```bash
objdump -d -C --no-show-raw-insn bin/vector_add | grep -E 'vaddps|vfmadd'
#   vaddps  ... %ymm0, %ymm0    <- 256-bit packed add
objdump -d --no-show-raw-insn bin/dot_product | grep -E 'vfmadd'
#   four vfmadd231ps ... %ymm{0,1,2,3} <- independent accumulators
```

`make disasm` does both and labels the functions.

## FMA numerics caveat

FMA computes `a*b + c` with a single rounding instead of two, so an
`avx2_fma` result is **not bit-identical** to the scalar `mul` + `add`
reference. The `saxpy` checksums differ in the last digits for that reason (the
default alpha is deliberately non-dyadic). The harness therefore compares
floating-point checksums with a relative tolerance of `1e-4`; never with `==`.
Integer kernels (`false_sharing_adjacent`, `false_sharing_padded`) are compared
exactly.

## Notes on the results

* `vector_add` and `saxpy` are memory-bound: the 4 Mi-element working set
  (48 MB) exceeds the 20 MB L3, so every variant streams from DRAM and the SIMD
  speedup is modest.
* `dot_product`/`sum_reduce` `autovec` show no speedup — the reduction order is
  preserved with scalar adds. `avx2_fma` wins by using four independent
  dependency chains.
* `mandelbrot` is compute-bound and branch-divergent; the OpenMP variant uses
  `schedule(dynamic, 4)` because the E-cores are much slower than the P-cores,
  so a static split would leave fast cores idle at the barrier.
* `false_sharing_adjacent` and `false_sharing_padded` give the same answer both
  ways (each thread touches only its own counter). The thread count is set by the
  `THREADS` env var, falling back to `omp_get_max_threads()` (so it honours
  `OMP_NUM_THREADS` and affinity); `n = iters * threads`. The counters are
  `volatile` so the increment cannot be folded into one add.
  `false_sharing_adjacent` keeps eight counters per cache line and is markedly
  slower than `false_sharing_padded`, where `alignas(64)` gives every counter
  its own line.
* Pin per variant class before measuring: `taskset -c 2` for the single-thread
  variants, and `taskset -c 0-11` with `OMP_NUM_THREADS` set for the OpenMP
  variants. Pinning a whole run to one core collapses the OpenMP variants to a
  single thread and their timings become meaningless. The host runs `powersave`
  with turbo enabled, so medians vary ~10-15% run-to-run; the harness always
  prints `min` and differences inside that band are ties, not effects.
