# Learning path

A suggested order that builds the story one layer at a time. Commands assume
your shell is in `parallel-computing-lab/`. Each step says what to look for, not
just what to type. Read `CONCEPTS.md` first for the vocabulary, then follow the
languages in the order below.

The through-line: start with C++ because it can give you a genuinely scalar
baseline and show you the generated assembly at every level; then see how the
same ideas surface in languages that automate (Rust), forbid (Go), or hide
(Python) the SIMD layer.

## 0. Orientation

```bash
lscpu                        # cores, SIMD flags, cache sizes
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # powersave here
```

Match what you see against `CONCEPTS.md`. Confirm AVX2 and FMA are in the
flags list and AVX-512 is not.

## 1. C++ — the full ladder

```bash
make -C cpp                  # builds bin/<kernel>, scalar..openmp
make -C cpp run              # every BENCH line, all 18 rows
```

Read the source in this order, then run the matching kernel:

```bash
# 1a. the honest scalar baseline
$EDITOR cpp/src/scalar_kernels.cpp      # compiled -fno-tree-vectorize
taskset -c 2 ./cpp/bin/vector_add --verbose
taskset -c 2 ./cpp/bin/dot_product --verbose

# 1b. what the compiler does to a plain loop
$EDITOR cpp/src/autovec_kernels.cpp     # compiled -march=native
make -C cpp vec-report

# 1c. hand-written SIMD
$EDITOR cpp/src/simd_kernels.cpp        # -mavx2 -mfma intrinsics
make -C cpp disasm
```

What to look for:

- `vec-report` shows `autovec` element-wise loops vectorized and the reduction
  loops **missed** (strict-FP reassociation). This is the single most important
  observation in the lab.
- `disasm` proves the intrinsics: `vaddps` for `vector_add`, four independent
  `vfmadd231ps %ymm{0,1,2,3}` for `dot_product`.
- Compare `scalar` vs `avx2_fma` for `vector_add` (a tie — memory-bound) against
  `dot_product` (~2.4× — compute plus ILP). Then open `results/p1_cpp_st.txt`
  and check the medians.

Single-thread versus multi-thread:

```bash
taskset -c 2    ./cpp/bin/mandelbrot --verbose            # scalar-ish, ~116 ms
taskset -c 0-11 env OMP_NUM_THREADS=12 ./cpp/bin/mandelbrot --verbose
```

The OpenMP number is ~10 ms if pinning is right. If it equals the scalar
number, you pinned the whole run to one core — the collapse described in
`METHODOLOGY.md`.

## 2. Rust — the same kernels, safer ISA handling

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path rust/Cargo.toml

taskset -c 2 ./rust/target/release/dot_product --verbose
objdump -d --no-show-raw-insn rust/target/release/dot_product | grep -c ymm
RAYON_NUM_THREADS=12 taskset -c 0-11 ./rust/target/release/dot_product --verbose
```

What to look for:

- In `rust/src/lib.rs`, the `avx2_fma` paths are `unsafe fn` under
  `#[target_feature(enable = "avx2,fma")]`, reached only after
  `is_x86_feature_detected!`. That runtime check plus a scalar fallback is the
  portable way to use intrinsics from stable Rust.
- Rebuild **without** `-C target-cpu=native` and compare `ymm` counts. The
  explicit `avx2_fma` path still works (per-function target feature); `autovec`
  drops to SSE2 at best.
- `std::simd` is nightly-only, so it is absent; the lab pins stable rustc
  1.96.1, which is why the code uses the stable `core::arch` intrinsics.

## 3. Go — parallelism without SIMD

```bash
make -C go
taskset -c 2    ./go/bin/vector_add               # scalar baseline
GOMAXPROCS=12 taskset -c 0-11 ./go/bin/dot_product
taskset -c 0-11 ./go/bin-tools/scale              # GOMAXPROCS 1..12 curve
```

What to look for:

- The gc compiler does not auto-vectorize (Go wiki, Compiler Optimizations), so
  there is no `autovec` variant. `GOAMD64=v3` only makes AVX2+FMA the baseline.
- `scale` is the clearest Amdahl/memory-ceiling picture in the lab: the curve
  flattens well below 12× because the reductions are bandwidth-bound.
- The experimental path exists to show what Go *could* do and why it is not
  ready:

  ```bash
  make -C go simd
  GOEXPERIMENT=simd GOAMD64=v3 ./go/bin-simd/vector_add
  ```

  Read the AVX→SSE transition-penalty note in `go/README.md` first; take scalar
  timings from the default build only.

## 4. Python — where the time actually goes

```bash
make -C python setup         # venv + numpy 2.5.2 / numba 0.67.0
make -C python lib           # builds native/libsimd.so with AVX2/FMA
cd python

taskset -c 2    .venv/bin/python bench.py vector_add
THREADS=12 taskset -c 0-11 .venv/bin/python bench.py dot_product
THREADS=12 taskset -c 0-11 .venv/bin/python bench.py sum_reduce
cd ..
```

What to look for:

- `pure_python` is ~1.1 s for `vector_add`; `numpy` is ~14 ms — roughly 77×.
  That gap is the interpreter, not SIMD.
- `numpy` ≈ `ctypes_simd`: both hit the AVX2 path (NumPy via NEP 38 dispatch,
  `ctypes` via the hand-written C). `objdump -d python/native/libsimd.so | grep
  -E 'vfmadd|vaddps'` confirms the C build.
- `multiprocessing` is the only true multi-core Python variant; it uses `fork`
  so the pre-created arrays are shared copy-on-write (no pickling).
- In the single-thread run, `multiprocessing` collapses to one core (a Pool
  timesharing one core) — expected, and discarded by merge precedence. The
  harness prints GIL state in its stderr context line; PEP 703/779 explain why
  a free-threaded build would behave differently.

## 5. Cross-language — the aggregate

```bash
./bench/run_all.sh
```

The harness rebuilds each lane, runs single-thread and multi-thread passes
pinned per variant class, enforces the class-based merge precedence, checks each
variant against its own language's `scalar`, writes `results/results.csv`, and
renders `docs/RESULTS.md`. `RESULTS.md` is generated from the real `BENCH,`
lines; read it together with `METHODOLOGY.md` and treat sub-15% differences as
ties.

## Experiments to try yourself

- **Thread count.** Rerun the reductions with `THREADS=6` on
  `taskset -c 0,2,4,6,8,10` versus `THREADS=12` on `taskset -c 0-11`. How much
  do the extra SMT threads actually add?
- **Include the E-cores.** Drop `taskset` and compare. Does wall time get worse,
  and does the variance grow? Why (P/E clock and width)?
- **Build target.** Rebuild the C++ `autovec` and Rust `autovec` paths for a
  lower baseline (`-march=x86-64`, `-march=x86-64-v2`, or no
  `-C target-cpu=native`). Watch `objdump` and the timings change together.
- **Alignment.** Change one `vector_add` kernel from `loadu`/`storeu` to
  `load`/`store` with `alignas(32)` buffers. Measure with `perf stat` to see
  whether it moved anything on this core.
- **Streaming stores.** Switch `vector_add`'s output stores to
  `_mm256_stream_ps` (+ `_mm_sfence`) and compare against the normal store. Does
  removing read-for-ownership traffic help a kernel that is already at the
  bandwidth limit?
- **Go baseline.** Rebuild with `GOAMD64=v1`, `v2`, `v3` and diff the scalar
  timings. If nothing changes, you have demonstrated "v3 does not
  auto-vectorize".
- **Fast-math.** Rebuild a reduction with `-ffast-math` (or Numba
  `fastmath=True`) and watch the optimizer vectorize a loop it previously
  refused — and the checksum drift.
- **Repetition count.** Run with `BENCH_REPS=1` and `BENCH_REPS=7`. The
  checksum must be identical; the median is what changes.
- **Size sweep.** Increase `--n` until every `vector_add` variant converges to
  the same time. That is the DRAM ceiling becoming visible; add `perf stat` to
  put cache-miss and bandwidth numbers on it.
