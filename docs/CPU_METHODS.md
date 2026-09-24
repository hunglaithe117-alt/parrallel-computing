# CPU methods: practical levers

This chapter is a toolbox. Each technique is tied to the kernel or build flag
that exercises it in the lab, so you can reproduce or break it yourself.

## Alignment and aliasing

An AVX2 register is 256 bits, so the natural alignment for a `float` array is
32 bytes. This matters for which intrinsic you may use:

- Aligned intrinsics (`_mm256_load_ps`, `_mm256_store_ps`) require a 32-byte
  aligned address; on a misaligned address they fault (or are undefined), so the
  compiler relies on the programmer's promise.
- Unaligned intrinsics (`_mm256_loadu_ps`, `_mm256_storeu_ps`) work at any
  address. On recent Intel cores an unaligned access that happens to be
  naturally aligned has no penalty; one that crosses a 64-byte cache line costs
  an extra cycle or two (Intel optimization manual).

The lab's buffers come from `bench::make_aligned` in C++ (`std::aligned_alloc`),
`Vec` in Rust, and slices in Go. The C++ buffers are 32-byte aligned, but the
AVX2 kernels still use `loadu`/`storeu`, because the kernel interface does not
promise alignment and an unaligned load is safe on any input. To use the aligned
intrinsics you must establish alignment yourself — `alignas(32)` (C++), an
aligned allocator, or an address assertion. `perf stat` is the only way to know
whether alignment actually changed anything on your data.

**`__restrict`** tells the compiler the three pointers do not alias, so it may
keep values in registers and reorder loads. In C++ the lab annotates every
kernel pointer with `__restrict` (see `cpp/src/kernels.h`). Rust does not need
it — `&[f32]` and `&mut [f32]` make non-overlap part of the type system — but Go
has no aliasing/`restrict` semantics at all, so its kernels rely on the harness's
sink to keep the work live.

**Tail handling.** `n` is rarely a multiple of 8. Every vector kernel processes
`n / 8` full vectors and then a scalar remainder loop for `n % 8`. The Python
lane's `native/simd.c` does the same. Masked loads/stores are the alternative;
the scalar tail is simpler and fast enough for ≤ 7 elements.

## Instruction-level parallelism

Latency, not bandwidth, limits a dependency chain. An FMA has a few cycles of
latency but the core can start several per cycle, so a loop that reuses one
accumulator stalls. Unroll and give each unrolled iteration its own
accumulator; the compiler can then overlap them.

The lab's reduction kernels use **four independent 256-bit accumulators**, i.e.
32 active `f32` partial sums, folded once at the end. That is the ILP that makes
`avx2_fma` `dot_product` ~2.4× the scalar reduction. The compiler will not do
this under strict FP because it requires reassociation; the programmer does it
explicitly. Unrolling alone (without independent accumulators) does not break
the chain.

## Cache blocking, prefetch, non-temporal stores

- **Blocking / tiling** raises temporal locality: process a tile that fits in
  L1/L2 before moving on. It is a large win for kernels with reuse (matrix
  multiply). The canonical lab kernels are element-wise or pure reductions with
  no reuse, so blocking would not help them; that is deliberate.
- **Software prefetch** (`_mm_prefetch(p, _MM_HINT_T0)`) issues a hint for data
  the hardware prefetcher may not predict. For the contiguous streams here the
  hardware prefetcher already does the job; software prefetch is for irregular
  traversals.
- **Non-temporal stores** (`_mm256_stream_ps`) write straight to memory,
  bypassing the cache and avoiding the read-for-ownership traffic a normal store
  incurs. They can raise streaming-store bandwidth noticeably, but require
  32-byte alignment and an `_mm_sfence()` before another thread reads the data.
  The lab keeps ordinary stores for clarity; switching `vector_add`'s output to
  streaming stores is a good experiment (see `LEARNING_PATH.md`).

## False sharing

Cache coherency works on 64-byte lines. Two threads that write *different*
variables on the *same* line invalidate each other's copy on every write — a
cache-line ping-pong that has nothing to do with the data being shared. The
result is correct and the timing is terrible, which makes it a classic
"parallelism made it slower" bug.

The lab demonstrates the fix directly:

- `false_sharing_adjacent`: counters packed in a `[]int64`, so eight counters
  share one 64-byte line.
- `false_sharing_padded`: `struct alignas(64) PaddedCounter { volatile long v; }`
  in C++; `int64 + [56]byte` (= 64 bytes total) in Go. One line per counter.

Measured (Phase 3, both at n = 24,000,000): Go `goroutines` 237.9 ms adjacent
vs 21.8 ms padded (~10.9×); C++ `openmp` 7.22 ms vs 1.53 ms (~4.7×). Compare
adjacent vs padded **within** a language, not C++ against Go: Go must use
`atomic.AddInt64` (a locked read-modify-write) where C++ uses a `volatile`
load/store, so the per-increment constant differs. Padding to **64 bytes**
matches this machine's line size. Some architectures use 128-byte lines or pair
adjacent lines with a spatial prefetcher; padding to 128 bytes can be safer in
those cases but wastes cache here.

## Thread count and placement

- **Count.** Prefer the physical set (`THREADS=6`, `taskset -c 0,2,4,6,8,10`,
  one thread per P-core) when you want a clean scaling curve, or the SMT set
  (`THREADS=12`, `taskset -c 0-11`) to match the contract. Always report which.
  SMT threads share a core's resources, so the 7th–12th threads add much less
  than the first six.
- **Affinity.** `taskset -c <list>` pins the process to a CPU set. It is
  mandatory here: pinning a whole run to one core collapses OpenMP / Rayon /
  goroutines / `multiprocessing` to a single worker, so the contract runs each
  binary twice (once per variant class). `taskset -pc $$` shows the current
  mask.
- **P versus E.** Exclude CPU 12–15 so a variant does not land on a slower,
  128-bit E-core. For uneven work, use dynamic scheduling (the C++
  `mandelbrot openmp` kernel does) so fast P-cores pick up more rows.
- **NUMA.** This host has a single socket and one NUMA node, so there is no
  remote-memory penalty and `numactl` is informational. On a multi-socket box,
  first-touch allocation and `numactl --localalloc`/`--cpunodebind` matter; the
  memory-bandwidth ceiling in `WHY_FASTER.md` becomes a per-socket ceiling.

## Compiler and runtime levers

| Tool | Lever | What it does |
|---|---|---|
| g++/gcc | `-O3 -march=native` | tunes for this CPU (AVX2+FMA here) |
| g++/gcc | `-mavx2 -mfma` | explicit baseline for the SIMD TUs |
| g++/gcc | `-fno-tree-vectorize -fno-tree-slp-vectorize` | disables auto-vectorization (the `scalar` reference) |
| g++/gcc | `-ffp-contract=off` | forbids FMA contraction, keeping scalar results comparable |
| g++/gcc | `-fopt-info-vec-optimized` / `-fopt-info-vec-missed` | report which loops vectorized and why the rest did not |
| clang | `-Rpass=loop-vectorize` / `-Rpass-missed=loop-vectorize` | the LLVM equivalent |
| Rust | `RUSTFLAGS="-C target-cpu=native"` | allows AVX2/FMA auto-vectorization |
| Rust | `#[target_feature(enable="avx2,fma")]` + `is_x86_feature_detected!` | per-function ISA enable with a runtime check |
| Rust | `f32::mul_add` | explicit FMA |
| Go | `GOAMD64=v3` | makes AVX2+FMA the baseline (does **not** auto-vectorize) |
| Go | `GOEXPERIMENT=simd` | experimental `simd/archsimd` (unstable, GA deferred to Go 1.28) |
| Python | NumPy | **NEP 38** runtime dispatch picks the AVX2 kernel |
| Python | `OPENBLAS_NUM_THREADS=1` etc. | keep NumPy single-threaded, as the contract requires |
| Python | Numba `@njit` | LLVM-compiles the loop; `parallel=True` for threads, `fastmath=True` changes results |

The gc compiler deliberately has **no auto-vectorizer** (Go wiki, Compiler
Optimizations), which is why the Go lane has no `autovec` variant. `GOAMD64=v3`
only makes the instructions legal, not automatic. Confirm what any compiler did
with `objdump -d ... | grep -E 'vaddps|vfmadd'` or the report flags above — never
assume a plain loop vectorized.

## Pitfalls

- **`-ffast-math` / `-Ofast`** permit reassociation and other IEEE-break
  transformations. They can change results and are excluded from the default
  build so the checksum gate stays meaningful. The same warning applies to Numba
  `fastmath=True`.
- **Assuming AVX-512 exists.** It does not on this hybrid client CPU, and code
  that assumes it will `SIGILL`. Guard with `__builtin_cpu_supports("avx512f")`
  or `is_x86_feature_detected!` and provide an AVX2 path.
- **Assuming aligned loads are safe.** `_mm256_load_ps` on an unaligned pointer
  is a crash or UB; use `loadu` unless you have proven alignment.
- **`-march=native` portability.** The binary is only valid on a CPU with at
  least these features. Build with an explicit target (`-march=x86-64-v3`) when
  the binary must run elsewhere.
- **Trusting the clock, not the work.** A benchmark that lets the compiler
  delete the kernel, or that measures a different amount of work per variant, is
  not a benchmark. That is what the contract's checksum and timing rules
  prevent (`METHODOLOGY.md`).
