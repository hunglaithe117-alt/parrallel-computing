# Python lane — pure Python vs NumPy vs multiprocessing vs ctypes→C SIMD

The Python lane shows the same kernels as the C++, Rust, and Go lanes, but from
Python's point of view: the cost of an interpreted element loop, the payoff from
NumPy's compiled ufuncs, the escape hatch from the GIL via `multiprocessing`, and
the bridge to a hand-written AVX2/FMA C library through `ctypes`.

It follows [`docs/BENCH_CONTRACT.md`](../docs/BENCH_CONTRACT.md) v2.1: one
`BENCH,` line per variant on stdout, everything else on stderr, median-of-`REPS`
timing, a mandatory checksum, and `n <= 0` rejected.

## Layout

```text
python/
├── Makefile          # setup / lib / run / clean
├── requirements.txt  # numpy==2.5.2, numba==0.67.0
├── bench.py          # contract harness: timing, median, min, BENCH, emission
├── kernels.py        # kernels + variant registry (pure/numpy/numba/mp/ctypes)
└── native/
    ├── simd.c        # scalar + AVX2/FMA C kernels
    └── libsimd.so    # built by `make lib` (gitignored)
```

## Quick start

```bash
make -C python setup      # create python/.venv and pip install -r requirements.txt
make -C python lib        # gcc -O3 -mavx2 -mfma -shared -fPIC -> native/libsimd.so
make -C python run        # all three examples, unpinned (convenience)

# Contract runs: pin to the P-cores (CPU 12-15 are E-cores — exclude them).
cd python
taskset -c 2    .venv/bin/python bench.py vector_add              # single-thread class
THREADS=12 taskset -c 0-11 .venv/bin/python bench.py dot_product  # multi-thread class
```

`bench.py` accepts `--n <count>` (overrides `BENCH_N`, which overrides the
per-example default) and `--reps <count>` (overrides `BENCH_REPS`, default 7).

## Per-example defaults

Defaults are chosen to bound **peak memory**, not to hit a wall-clock target:
the `pure_python` reference stores one boxed Python `float` per element (~32 B in
a list), so `n` is the dominant RAM term. Every default can be raised with
`--n`/`BENCH_N`; peak RAM then scales roughly linearly (`n * 32 B` per input
list, plus `n * 4 B` per `float32` array).

| example | default `n` | measured `pure_python` | peak RSS (ST run) |
|---|---|---|---|
| `vector_add` | 8 000 000 | ~0.30 s | ~1.30 GiB |
| `dot_product` | 8 000 000 | ~0.36 s | ~0.88 GiB |
| `sum_reduce` | 16 000 000 | ~0.28 s | ~0.86 GiB |

The earlier 32 M / 32 M / 56 M counts put the reference near 1 s but peaked at
several GB of boxed floats; they remain reachable via `--n 32000000` for a
machine with the headroom. Peak RSS was measured with `/usr/bin/time -v`.

## Variants

| variant | class (contract v2.1) | what it does |
|---|---|---|
| `pure_python` | single-thread | lists + explicit loop; the within-language reference |
| `numpy` | single-thread | ufunc on contiguous `float32` arrays |
| `numba` | single-thread | `@njit` compiled loop (optional) |
| `multiprocessing` | multi-thread | `fork` `Pool(THREADS)` over disjoint chunks |
| `ctypes_scalar` | single-thread | scalar C loops in `libsimd.so` |
| `ctypes_simd` | single-thread | AVX2/FMA C loops in `libsimd.so` |

`multiprocessing` is the only multi-core variant in this lane. `ctypes_simd` is a
single AVX2 call and is classified **single-thread** by contract v2.1, so its row
comes from the single-thread run.

All variants use the same deterministic positive `float32` data per example, so
checksums are comparable within the lane. The harness compares every variant's
checksum against `pure_python` with the contract's true relative tolerance of
`1e-4` (denominator `max(|ref|, 1e-12)`) and exits non-zero on a failure.

## Why NumPy is fast

* **Contiguous, typed buffers.** A `numpy.ndarray` is a pointer plus a dtype,
  shape, and strides over one flat C buffer. The inner loop is compiled C over
  `float*` with no per-element boxing/unboxing, unlike a Python list of `float`
  objects (each element is a heap object plus a pointer). `flags="C_CONTIGUOUS"`
  on the ctypes side guarantees the same layout reaches the C kernels.
* **Ufuncs and the GIL.** A ufunc (`np.add`, `np.dot`, …) runs its typed inner
  loop in C and releases the GIL for the duration. Releasing the GIL lets other
  Python threads make progress, but element-wise ufuncs are memory-bandwidth
  bound, so free-threading alone rarely speeds them up; you need task
  parallelism (below). For `vector_add`, only the array operation is timed
  (`np.add(a, b, out=c)`), and the checksum is a separate untimed pass.
* **NEP 38 runtime SIMD dispatch.** NumPy ships the same ufunc inner loops
  compiled several times for different CPU baselines (SSE2/SSE4/AVX2/AVX-512)
  and selects an implementation at import time from the detected CPU features.
  On this machine (AVX2 + FMA, no AVX-512) it picks the AVX2/FMA kernels, so
  `numpy` and `ctypes_simd` land within a few percent of each other.

## GIL vs `multiprocessing`

A default CPython build has a Global Interpreter Lock: only one thread executes
Python bytecode at a time. CPU-bound Python threads therefore serialize, so
`threading` cannot parallelize `vector_add`, `dot_product`, or `sum_reduce`.
`multiprocessing` sidesteps this with separate interpreters and address spaces:

* The input arrays are created *before* the pool is forked, so Linux
  copy-on-write inheritance hands every worker the same pages with no pickling.
* The pool size is the effective thread count: `THREADS` when set, otherwise the
  CPU affinity mask (`len(os.sched_getaffinity(0))`), clamped to that mask. A
  `taskset -c 2` run therefore resolves to **1** worker and the
  `multiprocessing` variant is skipped with a stderr note — forking twelve
  processes onto one core measures oversubscription, not parallelism.
* The pool is forked *before* Numba's LLVM compile probes run, so no LLVM
  threads exist at fork time.
* `vector_add` must write an output the parent can see, so its output lives in
  `multiprocessing.shared_memory`; the reductions return small partial scalars
  that the parent sums (included in the timed region — the reduction *is* the
  kernel).
* `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/… are forced to 1 before NumPy is
  imported: NumPy is a single-thread variant, and forking a process whose BLAS
  has already spawned threads is unsafe.
* The `fork` start method is requested explicitly. Python 3.14's *default* is now
  `forkserver`, which would not inherit the arrays, so the explicit `fork`
  context is part of the design on Linux.

The validation run shows the point: `dot_product` with `THREADS=12` on
`taskset -c 0-11` runs in a few ms with `multiprocessing` versus ~0.3 s in
`pure_python`. In the single-thread run (`taskset -c 2`, affinity = 1) the
variant is skipped rather than forked. An earlier reading of ~39 ms in that run
was twelve forked processes timesharing one core — oversubscription, not a
collapse to one thread.

## Free-threading status

* This default build reports `gil=on` via `sys._is_gil_enabled()` (added in
  Python 3.13); `bench.py` prints that status in its stderr context line.
* Free-threaded CPython (PEP 703) makes the GIL optional; PEP 779 defines the
  criteria for treating that build as supported, which CPython 3.14 meets. On a
  free-threaded build, `sys._is_gil_enabled()` returns `False` and pure-Python
  threads can run in parallel.
* The lane does not depend on which build it runs under: `multiprocessing`
  parallelizes on both, and the checksums are unaffected. The harness simply
  reports the build's GIL state so a reader can interpret the numbers.

## numba is optional

`numba==0.67.0` is pinned in `requirements.txt` and is used by the `numba`
variant (`@njit` loops compiled with LLVM). It is treated as optional at every
layer:

* `make setup` installs the pins; if the full set cannot be resolved it retries
  `numpy==2.5.2` alone and continues, so the lane still runs.
* `kernels.py` imports Numba defensively and probe-compiles each kernel on a
  tiny slice during setup (untimed). If the import or the compile fails, it logs
  the reason to stderr and emits **no** `BENCH,` line for `numba`; the other
  variants run unaffected.
* `bench.py` also catches a failure in any single variant, logs it, and keeps
  going.

The same skip-and-continue behaviour applies to the ctypes variants when
`native/libsimd.so` has not been built: stderr says to run `make lib`, and the
run proceeds with `pure_python`, `numpy`, `numba`, and `multiprocessing`.

## The ctypes bridge

`native/simd.c` implements scalar and AVX2/FMA versions of all three kernels. It
is built exactly as the task specifies:

```bash
gcc -O3 -mavx2 -mfma -shared -fPIC -o native/libsimd.so native/simd.c
```

* The AVX2 kernels use `_mm256_loadu_ps`/`_mm256_storeu_ps` and
  `_mm256_fmadd_ps`, with a scalar tail for `n % 8`. `dot_product` and
  `sum_reduce` keep four independent accumulators to expose instruction-level
  parallelism, then fold the eight lanes once.
* The scalar functions are decorated with
  `__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize",
  "fp-contract=off")))`, so despite the file being compiled `-mavx2 -mfma`,
  `ctypes_scalar` stays genuinely scalar (only scalar `vaddss`, no packed
  `vaddps`).
* `kernels.py` loads it with `ctypes.CDLL` and declares every signature with
  `numpy.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")`
  plus `c_size_t`/`c_double`. The whole array is passed with **one call per
  kernel invocation** — never per element — and the buffers are allocated before
  timing.
* The AVX2 reduction accumulators are 256-bit `float32`; only the horizontal
  fold (`hsum_ps`) and the scalar tail accumulate in `double`. That float32 lane
  accumulation — not a double vector accumulator — is why the C checksums differ
  from the `float64` `pure_python` reference by roughly `1e-6`, well inside the
  contract's `1e-4` relative tolerance.

Prove the AVX2 code is really there:

```bash
objdump -d python/native/libsimd.so | grep -E 'vfmadd|vaddps'
# 16 vaddps, 5 vfmadd231ps (packed 256-bit), 3 vfmadd231sd (scalar tails)
```

## Pinning, `THREADS=12`, and the hybrid CPU

CPU 0-11 are **6 physical P-cores × 2 SMT threads** (e.g. CPU 2 and 3 share a
core); CPU 12-15 are E-cores and are excluded.

* Single-thread class: `taskset -c 2`.
* Multi-thread class: `THREADS=12 taskset -c 0-11`. `bench.py` reads `THREADS`,
  falls back to the CPU affinity count when it is unset, and clamps it to the
  affinity mask when it is larger — so the Python lane uses the same thread count
  as the other lanes. The contract's alternative is
  `THREADS=6 taskset -c 0,2,4,6,8,10` (one thread per physical P-core).
* Never pin a multi-thread run to one core and read `multiprocessing` from it:
  that run is for the single-thread class only (merge precedence).

## Contract compliance

| Contract rule | Where |
|---|---|
| `time.perf_counter_ns`, one warm-up | `bench.py:measure` |
| median of `BENCH_REPS` (even count = mean of two middle) | `bench.py:measure` |
| `--n` > `BENCH_N` > per-example default; `n <= 0` rejected | `bench.py:main`, `positive_int` |
| exactly `BENCH,<example>,<variant>,<n>,<ms>,<checksum>` on stdout | `bench.py:main` |
| unconditional `min` + context on stderr | `bench.py:main` |
| element-wise checksum in a separate untimed pass | `kernels.py` `Variant.checksum` |
| reduction checksum is the timed kernel | `kernels.py` reductions |
| checksum sink defeats dead-code elimination | `bench.py` `_SINK`/`consume` |
| within-language tolerance vs `pure_python` (`1e-4` relative) | `bench.py:main` |
| skip a broken optional variant without aborting | `kernels.py` probe/skip, `bench.py` catch |

Checksums describe a single kernel application and are **REPS-invariant**:
element-wise kernels sum their output once after timing, and reductions return
the same deterministic scalar on every rep. Verified with `BENCH_REPS=1` vs
`BENCH_REPS=7` (all variants byte-identical; see the validation note below).

## Measured on this machine (validation run)

`vector_add`, pinned `taskset -c 2` (single-thread class), n=8 000 000, reps=7
(median ms). The `multiprocessing` row is absent because the affinity mask holds
one CPU:

| variant | ms |
|---|---|
| pure_python | 296.14 |
| numpy | 3.89 |
| numba | 3.56 |
| ctypes_scalar | 4.20 |
| ctypes_simd | 3.82 |

`dot_product`, single-thread (`taskset -c 2`) vs multi-thread
(`THREADS=12 taskset -c 0-11`), n=8 000 000, reps=7 (median ms):

| variant | ST ms | MT ms |
|---|---|---|
| pure_python | 357.38 | 375.75 |
| numpy | 2.01 | 1.97 |
| multiprocessing | skipped | 1.69 |
| numba | 4.24 | 4.27 |
| ctypes_scalar | 6.22 | 4.28 |
| ctypes_simd | 2.32 | 1.95 |

Per the contract, `min` is the stability reference and differences under ~15%
are ties: treat NumPy / `ctypes_simd` / `multiprocessing` in the tables above as
within one noise band rather than a strict ordering. Raw `BENCH,` lines and
stderr context for these runs are reproduced in the task report.
