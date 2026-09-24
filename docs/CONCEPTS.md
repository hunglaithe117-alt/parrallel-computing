# Concepts: SIMD, cores, and memory

This lab measures the same handful of kernels written four ways. Before looking
at any number, it helps to have a model of *where* the speed comes from. This
chapter covers the hardware facts that the rest of the documentation assumes:
vector width, core types, caches, and the memory wall. Machine-specific values
come from `lscpu` on the benchmark host.

## What SIMD is

Ordinary scalar code issues one instruction per value: `a[i] + b[i]` produces
one `f32` per `add`. **SIMD** (Single Instruction, Multiple Data) packs several
values into one wide register and does the same operation to all of them with
one instruction (Intel SDM calls these *packed* operations). The ALU work is
widened; the instruction stream is not.

Two consequences are worth stating up front:

- SIMD multiplies *arithmetic* throughput, not memory bandwidth. A vector load
  fetches the same number of bytes a scalar load does; it just moves more of
  them per instruction. A kernel that is waiting on DRAM stays waiting.
- SIMD only helps when the same operation applies to every lane. Branchy,
  data-dependent work (the Mandelbrot escape loop) needs masking or predication,
  and lanes that diverge do idled work.

## x86 vector extensions

| Extension | Register | Width | f32 lanes | f64 lanes | i32 lanes |
|---|---|---|---|---|---|
| SSE2 (baseline of x86-64) | `xmm` | 128-bit | 4 | 2 | 4 |
| AVX2 | `ymm` | 256-bit | 8 | 4 | 8 |
| AVX-512 | `zmm` | 512-bit | 16 | 8 | 16 |
| FMA3 | same as SSE/AVX | — | — | — | — |

SSE2 is guaranteed on every x86-64 CPU (Intel SDM, `CPUID` baseline). AVX2 and
FMA3 are extensions this host has. AVX-512 is listed only for context: it is
**not** available here.

**FMA** (fused multiply-add) is orthogonal to width. A normal multiply-then-add
rounds twice (once after the multiply, once after the add); an FMA computes
`a*b + c` with a **single rounding** (Intel SDM, `VFMADD*`). On AVX2 that is
`vfmadd231ps` over a `ymm` register: eight `f32` multiply-adds per instruction.
The one-rounding difference is why FMA kernels are never bit-identical to a
scalar `mul` + `add` reference, and why this lab compares floating-point
checksums with a relative tolerance rather than `==` (see `METHODOLOGY.md`).

## Why width and FMA raise the ceiling

For a compute-bound loop with no dependency limit, the arithmetic ceiling scales
with the number of lanes: 4× `f32` per SSE2 instruction, 8× per AVX2
instruction, and FMA fuses a multiply and an add into one. Going from scalar
(1 element/instr) to AVX2+FMA (8 elements/instr, mul+add fused) is therefore a
potential order-of-magnitude change in *instructions retired per element*.

That ceiling is only reachable if the loop exposes enough independent work. A
single running accumulator is a serial dependency chain — each add waits on the
previous one — so vectorizing it alone buys little. Reductions in this lab
therefore keep **several independent vector accumulators** and fold them at the
end, a technique covered in `WHY_FASTER.md` and `CPU_METHODS.md`.

## This machine

- **CPU**: Intel Core i5-14400 (Raptor Lake, hybrid). 10 cores, 16 hardware
  threads.
- **SIMD available**: SSE2, AVX2 (256-bit), FMA3, AVX-VNNI. **No AVX-512.**
- **Compilers**: gcc/g++ 15.2, clang 23.1, rustc 1.96.1 (stable), go 1.27.1,
  CPython 3.14.7 + NumPy 2.5.2 / Numba 0.67.0.

Raptor Lake is a *hybrid* design: fast P-cores plus efficient E-cores. The
E-cores (Gracemont) implement a **128-bit** vector datapath and have **no
AVX-512**. Rather than let the two core types support different instruction
sets, Intel fuses off AVX-512 on the whole package on these hybrid client parts
(Intel's hybrid architecture documentation). The practical result: on this CPU
the widest portable vector is 256-bit AVX2, and every SIMD path in the lab stops
there. Do not assume AVX-512 exists in code you write for this machine.

## P-cores, E-cores, and SMT

`lscpu` reports 16 CPUs over 10 cores. The contract uses CPU 0–11 as
**6 physical P-cores × 2 SMT threads** (for example CPU 2 and CPU 3 share one
physical core) and treats CPU 12–15 as the **4 E-cores**.

| Property | P-core | E-core |
|---|---|---|
| Max clock (this host) | 4700 MHz | 3500 MHz |
| Vector datapath | 256-bit | 128-bit (Gracemont) |
| SMT | yes (2 threads) | no |
| L1d | 48 KiB | 32 KiB |
| Share of L2 | 1.25 MiB private | shared cluster |

Consequences for benchmarking:

- An E-core is slower on *both* clock and vector width. An AVX2 instruction on
  an E-core internally occupies a 128-bit datapath for more cycles. The contract
  excludes CPU 12–15 so a variant is not penalised for landing on an E-core.
- **SMT** lets two threads share one P-core's execution resources. It usually
  helps throughput under memory stalls, but 12 SMT threads will not deliver
  12× the work of one. The contract states physical-vs-logical explicitly
  (`THREADS=12` with SMT, or `THREADS=6` on the physical set).
- When work is uneven, the scheduler must not hand a slow E-core a large static
  share. The C++ `mandelbrot` OpenMP kernel uses `schedule(dynamic, 4)` for
  exactly this reason.

## Cache hierarchy

Caches are why "fits in cache" is a real performance category, not a slogan.
On this host `lscpu` reports:

| Level | Size | Scope |
|---|---|---|
| L1d | 48 KiB (P) / 32 KiB (E) | per core |
| L2 | 1.25 MiB (P) | per P-core; E-cores share a cluster L2 |
| L3 | 20 MiB | shared across the package |
| Cache line | 64 bytes | all levels |

Latency grows sharply with distance. Approximate orders of magnitude on a
Golden-Cove/Raptor-Cove-class core (Intel 64 and IA-32 Optimization Reference
Manual): L1 hits are a handful of cycles, L2 roughly a dozen, L3 a few dozen,
and a DRAM miss a few hundred. Exact counts vary; treat these as ballpark.

The working sets the cross-language harness actually uses (all `f32`, harness
`n = 4,000,000`; each lane's own default can be larger):

| Kernel | Bytes touched | Versus 20 MiB L3 |
|---|---|---|
| `vector_add` | 48 MB (a, b, out) | exceeds |
| `saxpy` | 48 MB (read x, read + write y) | exceeds |
| `dot_product` | 32 MB (a, b) | exceeds |
| `sum_reduce` | 16 MB (one array) | fits |
| `mandelbrot` output | ~0.77 MB | fits |

`vector_add`, `saxpy`, and `dot_product` exceed L3 and stream from DRAM;
`mandelbrot`'s output sits in cache and the kernel is compute-bound.
`sum_reduce`'s single 16 MB array fits, but a one-pass read has no reuse, so it
is still bandwidth-limited. That single distinction explains most of the results
later in these docs.

## Memory bandwidth versus compute

The speed limit of a loop is whichever resource runs out first. The standard
tool is **arithmetic intensity**: floating-point operations per byte moved.

| Kernel | FLOPs per element | Bytes per element | Intensity (FLOP/byte) |
|---|---|---|---|
| `vector_add` `c=a+b` | 1 | 12 (read a, b, write c) | ~0.08 |
| `saxpy` `y=ax+y` | 2 | 12 (read x, y, write y) | ~0.17 |
| `dot_product` | 2 | 8 (read a, b) | ~0.25 |
| `sum_reduce` | 1 | 4 (read a) | ~0.25 |
| `mandelbrot` | many | ~1 output byte | high |

Two caveats. First, a normal store is **write-allocate**: the core reads the
line it is about to overwrite (read-for-ownership), so a streaming store can
cost extra hidden traffic; non-temporal stores avoid it (`CPU_METHODS.md`).
Second, "bytes" here counts logical traffic, not DRAM transactions; `perf stat`
is the way to see the real memory traffic.

The **Roofline** model (Williams, Waterman, Patterson, CACM 2009) plots
attainable FLOP/s as `min(compute_peak, bandwidth × intensity)`. Kernels left of
the "ridge" are memory-bound: no amount of extra compute width helps, only
more bandwidth. On a machine of this class the ridge sits at a few FLOP/byte,
and every element-wise or reduction kernel in the table is far below it. That
is the one-sentence explanation for the result that most surprises newcomers:
at large `n`, `scalar`, `sse2`, and `avx2_fma` `vector_add` finish within a few
percent of one another (see `WHY_FASTER.md`). They are all waiting on the same
DRAM.

To watch the ceiling for yourself, sweep `n` upward on `vector_add` until the
`scalar`/`avx2_fma` gap closes and `perf stat` reports a high cache-miss rate:
that is the DRAM limit. The exact bytes-per-second figure is machine- and
DIMM-dependent, so it is deliberately not asserted here.
