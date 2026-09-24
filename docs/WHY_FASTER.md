# Why faster: the sources of speedup

This chapter connects each source of speedup to a kernel in the lab and to
measured numbers. Unless stated otherwise, numbers are **median milliseconds
from the pinned Phase 1/2 runs** in `results/` on this machine. The host runs
the `powersave` governor with turbo enabled, so medians vary ~10–15% run to run;
differences under ~15% are **ties**, not effects (`METHODOLOGY.md`).

## SIMD width: scalar → SSE2 → AVX2/FMA

The arithmetic ceiling scales with lanes: 1 `f32` per scalar instruction, 4 with
SSE2, 8 with AVX2, and FMA fuses a multiply and an add into one instruction with
a single rounding (Intel SDM). Where it shows up:

- `mandelbrot` (C++, n = 786,432 pixels) evaluates **8 pixels per iteration**
  with AVX2 and lane masks: `scalar` 116.4 ms → `avx2_fma` 16.5 ms (~7×). This
  is the cleanest case: the set is in cache and the work is arithmetic, so the
  extra lanes convert almost directly into speed.
- `mandelbrot` does **not** auto-vectorize, because its inner escape loop has a
  data-dependent trip count and branch. `autovec` ≈ `scalar` (Rust: 35.4 vs
  35.8 ms) — a genuine result, not a missing case.

Where it does *not* show up is just as important:

- `vector_add` (C++, n = 4.19 M): `scalar` 2.26, `sse2` 2.01, `avx2_fma` 1.90 ms.
  All three are inside the noise band around a common DRAM ceiling, so the honest
  reading is "no measurable SIMD win here", not "SIMD is broken".

## Reduction with multiple accumulators (ILP)

A reduction has a serial dependency: each partial sum waits on the previous one.
Vectorizing a single accumulator leaves most lanes idle during the latency of
the add, so it buys little. The lab's `dot_product` / `sum_reduce` kernels use
**four independent vector accumulators**, each covering 8 `f32` lanes, and fold
them once at the end. That exposes instruction-level parallelism (ILP) across
four independent dependency chains.

- C++ `dot_product` (n = 4.19 M): `scalar` 2.24 → `avx2_fma` 0.92 ms (~2.4×).
- C++ `sum_reduce` (n = 16.8 M): `scalar` 8.67 → `avx2_fma` 2.48 ms (~3.5×).
- Rust `dot_product` (n = 8 M): 4.97 → 2.19 ms; `sum_reduce`: 4.13 → 1.26 ms.

`autovec` ties with `scalar` for both reductions in both lanes (C++ 2.18 / 8.65;
Rust 4.94 / 4.13), for the reason in the next section.

## Auto-vectorization versus hand-written intrinsics

Element-wise loops vectorize easily under strict IEEE floating point: no
operation is reordered. Reductions are different. `sum += a[i]*b[i]` is a
non-reassociable chain, and neither LLVM (Rust) nor GCC (C++) is allowed to
break it without `-ffast-math`. The Rust lane demonstrated this directly by
compiling two loop shapes with `black_box(len)`:

| Loop shape | YMM instructions | Arithmetic |
|---|---|---|
| element-wise `out[i] = a[i] + b[i]` | 12 | `vaddps` (vectorized) |
| reduction `s += f64::from(a[i]) * f64::from(b[i])` | 0 | `vmulsd` (scalar) |

So `scalar ≈ autovec` for reductions is expected. Hand-written intrinsics
(`core::arch` in Rust, `<immintrin.h>` in C) let the programmer make the
reassociation and FMA-contraction decisions the compiler will not make under
strict FP. That is the only reason `avx2_fma` wins there.

One contract subtlety: C++ `scalar` is compiled with `-fno-tree-vectorize`, so
it really is scalar machine code. Rust and Go `scalar` are the language's
reference loop and may still be auto-vectorized — the contract says `scalar` is
a **within-language reference, not a cross-language baseline**.

## Cache behavior and prefetching

Caches turn a latency problem into a locality problem. When the working set
exceeds the 20 MiB L3, every variant streams from DRAM and SIMD width no longer
matters; the numbers above (`vector_add`, `sum_reduce`) are exactly that regime.
When the set fits in cache, the compute width is the ceiling (`mandelbrot`).

Hardware prefetchers recognize sequential streams and hide DRAM latency for the
element-wise kernels, which is much of why they approach the bandwidth limit
without explicit tuning. **Software prefetch** (`_mm_prefetch`) is for
irregular or pointer-chasing access the hardware cannot predict; it is not
needed for these contiguous kernels. **Cache blocking** raises reuse for
matrix-style kernels, which the canonical lab kernels deliberately do not have.

## Parallelism

All four languages express the same decomposition: split the index range, have
each worker compute a partial, combine the partials.

- C++: OpenMP (`#pragma omp parallel for`, `schedule(dynamic, 4)` for
  `mandelbrot`).
- Rust: Rayon's work-stealing pool.
- Go: a goroutine worker pool over contiguous chunks, one slot per worker in a
  preallocated partials slice.
- Python: `multiprocessing` with a `fork` pool over disjoint chunks; the input
  arrays are created before the fork so copy-on-write shares them without
  pickling.

**Amdahl's law** is the pessimistic view: a fixed serial fraction `(1-p)`
limits speedup to `1 / ((1-p) + p/N)`. **Gustafson's law** is the scaled view:
if the problem grows with the machine, the serial fraction matters less. Both
are useful here. Compute-bound `mandelbrot` scales nearly ideally — C++
`openmp` 10.3 ms vs `scalar` 116.4 ms (**~11.3×**) — because it has almost no
serial work and its working set stays in cache.

The memory-bound reductions do **not** get 12× on 12 SMT threads:

| Kernel | scalar | parallel | speedup |
|---|---|---|---|
| Go `dot_product` (n = 4.19 M) | 4.17 ms | 0.69 ms | ~6.0× |
| Go `sum_reduce` (n = 16.8 M) | 8.67 ms | 1.38 ms | ~6.3× |
| Rust `sum_reduce` (n = 8 M) | 4.13 ms | 0.50 ms | ~8.3× |
| Rust `dot_product` (n = 8 M) | 4.97 ms | 1.25 ms | ~4.0× |

The reason: every thread pulls from the same DRAM bus. Once the stream is
bandwidth-saturated, adding threads cannot create bandwidth. SMT sharing on the
P-cores compounds it. This is the **memory-bandwidth ceiling**, and it is why
"12 threads" so often buys 5–6× rather than 12×.

A different kind of parallel trap is **false sharing**. `false_sharing_adjacent`
keeps eight 8-byte counters on one 64-byte cache line, so the cores ping-pong
the line on every increment; `false_sharing_padded` gives each counter its own
line. Same arithmetic, same result. Phase 3, both at n = 24,000,000:

- Go (`goroutines`): adjacent 237.9 ms vs padded 21.8 ms (~10.9×).
- C++ (`openmp`): adjacent 7.22 ms vs padded 1.53 ms (~4.7×).

Compare adjacent vs padded **within** a language, not C++ against Go: Go must
use `atomic.AddInt64` (a locked read-modify-write) while C++ uses a `volatile`
load/store, so the per-increment constant differs.

## Why Python is different

Three mechanisms explain the Python numbers; none of them is "Python got
faster".

**The GIL.** A default CPython build executes only one thread of Python
bytecode at a time, so CPU-bound `threading` serializes. `multiprocessing`
sidesteps this with separate interpreters and address spaces; that is why it is
the genuine multi-core variant in the lane. PEP 703 makes the GIL optional and
PEP 779 defines when a free-threaded build is considered supported; this build
reports `gil=on` via `sys._is_gil_enabled()`.

**Library vectorization.** NumPy's ufuncs run compiled C inner loops and use
**NEP 38** runtime dispatch to select an SSE2/AVX2/AVX-512 build of the loop
matching the detected CPU. On this host that is the AVX2/FMA path, so `numpy`
and the hand-written `ctypes_simd` land within a few percent.

**JIT.** Numba (`@njit`) compiles the loop with LLVM. For memory-bound
element-wise work it ties NumPy: `vector_add` n = 32 M, `numpy` 14.23 vs
`numba` 14.56 ms. For the reduction it is slower: `dot_product` `numpy` 8.23 vs
`numba` 15.26 ms. The likely cause is that a plain `@njit` loop keeps a scalar
`float64` accumulator and is not turned into a multi-accumulator SIMD reduction;
that interpretation is plausible but not directly proven by this lab, so treat
it as a hypothesis.

The headline gaps are interpreter versus compiled code, not f32 versus f64:

| Kernel | pure Python | best compiled variant |
|---|---|---|
| `vector_add` (n = 32 M) | 1098.7 ms | `numpy` 14.23 ms (~77×) |
| `dot_product` (n = 32 M) | 1381.7 ms | `multiprocessing` 7.36 ms (~188×) |
| `sum_reduce` (n = 56 M) | 902.2 ms | `multiprocessing` 7.29 ms (~124×) |

## What the numbers mean

- **Memory-bound kernels** show flat timing per byte as `n` grows, and width
  gains vanish. `vector_add`, `saxpy`, and large-N `sum_reduce` are all here.
  A tie between `scalar` and `avx2_fma` is the expected signal.
- **Compute-bound kernels** scale with operations and with lanes.
  `mandelbrot` is the lab's example.
- **Ties are results.** `numpy` / `ctypes_simd` / `multiprocessing` in the
  Python tables can sit within one noise band; presenting them as a strict
  ranking would be reading noise. The harness flags any difference under ~15%
  as a tie.
- **A variant can win for the wrong reason.** Comparing checksums inside the
  timed region, using a different accumulator count, or not pinning the run can
  manufacture a speedup that is not real. The benchmark contract exists to
  remove those; `METHODOLOGY.md` explains how.
