# Cross-language benchmark harness

`run_all.sh` builds all four language lanes, runs every benchmark pinned as a
single-thread pass and a multi-thread pass, captures **only** the stdout
`BENCH,` lines, and regenerates `results/results.csv` and `docs/RESULTS.md`
from those captures. It implements the aggregation rules of
`docs/BENCH_CONTRACT.md` (v2.1); it does not modify the language folders.

## Usage

```sh
bash bench/run_all.sh                       # full run (THREADS=12)
bash bench/run_all.sh --quick               # smoke run: n = 200000, REPS = 3
THREADS=6 RUN_TAG=t6 bash bench/run_all.sh  # physical-P-core run, tagged outputs
bash bench/scaling.sh                       # thread-scaling curve -> results/scaling.csv
```

The script resolves the lab root from its own path, so it can be invoked from
any working directory. It is idempotent: every capture directory and generated
file is recreated on each run. Progress goes to stderr; stdout stays empty.

## What it does

1. **Builds** each lane (non-fatally):
   - C++: `make -C cpp`
   - Rust: `RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path rust/Cargo.toml`
   - Go: `make -C go build` (`GOAMD64=v3`, binaries in `go/bin/`)
   - Python: `make -C python setup` if `python/.venv` is missing, then
     `make -C python lib`
   - Each outcome is recorded in `results/raw/build_status.txt`
     (`BUILD,<lang>,ok|FAIL`). A failed build does not stop the run, but the
     report's "Build / capture warnings" section names the lane and its build
     log, so a stale-binary result is never silently presented as fresh.
2. **Runs every example twice**, capturing stdout to
   `results/raw/raw_<lang>_<pass>.txt` and stderr to `results/raw/logs/`
   (or `results/raw_<tag>/` when `RUN_TAG` is set):
   - ST pass: `taskset -c 2` (one P-core).
   - MT pass: `THREADS=… OMP_NUM_THREADS=… RAYON_NUM_THREADS=… GOMAXPROCS=…`
     (all equal to `THREADS`) with the THREADS-derived P-core set (see
     [THREADS-aware affinity](#threads-aware-affinity) below).
   - `BENCH_REPS` is exported from `REPS` for every run.
   - Cross-language kernels get an explicit, identical `n`:
     `vector_add`, `saxpy`, `dot_product`, `sum_reduce` = `4000000` (full) /
     `200000` (`--quick`). `mandelbrot` and `false_sharing_*` use each lane's
     default size because they are excluded from cross-language ranking.
   - C++ takes `--n`, Rust and Go take `BENCH_N`, Python takes
     `python/bench.py <ex> --n`.
   - A missing binary or an example that emits no `BENCH,` line is recorded in
     `results/raw/skips.txt` instead of failing the run (e.g. Go/Python have no
     `saxpy`).
3. **Aggregates** with `aggregate.py`:
   - merge precedence: single-thread variant rows are taken only from the ST
     pass, multi-thread rows only from the MT pass;
   - within-language correctness: every variant vs that language's own
     `scalar` (Python: `pure_python`) at the same `(example, n)`, relative
     tolerance `1e-4`, denominator `max(|reference|, 1e-12)`;
   - a cross-language kernel whose per-language `n` diverge is a **hard error**:
     aggregation refuses to write `results.csv`/`RESULTS.md` rather than emit an
     `n = mixed` table;
   - emits a **Build / capture warnings** section: failed lane builds (from
     `results/raw/build_status.txt` or `build_<lang>.log`) and any per-pass
     capture that is short of the variants that example should emit;
   - writes `results/results.csv` (`language,example,variant,class,n,ms,checksum,status`);
   - renders `docs/RESULTS.md` from the merged rows (never hand-written
     numbers).

Exit status is non-zero if any correctness check `FAIL`s; the Python lane's own
internal gate only fails its own process, the aggregator is the authority.

## Flags and environment

| Variable | Default | Meaning |
|---|---|---|
| `THREADS` | `12` | Multi-thread pool size (`OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `GOMAXPROCS`); the MT CPU set derives from it |
| `P_SET` | _(derived)_ | Explicit override for the MT `taskset` CPU set; default derives from `THREADS` |
| `ST_CPU` | `2` | `taskset` CPU for the single-thread pass |
| `REPS` | `7` | Timed repetitions, forwarded as `BENCH_REPS` |
| `RUN_TAG` | _(empty)_ | Tags outputs: `results/raw_<tag>/`, `results/results_<tag>.csv`, `docs/RESULTS_<tag>.md`; empty keeps the default paths |
| `PYTHON_BIN` | `python3` | Interpreter for `aggregate.py` (stdlib only) |

`--quick` sets `REPS` to `3` and the cross-language `n` to `200000` unless
those are overridden in the environment.

## THREADS-aware affinity

The multi-thread CPU set derives from `THREADS` (`bench/affinity.sh`,
`pcore_set`); an explicit `P_SET` in the environment still wins. Primary
P-core logical CPUs on this box are 0,2,4,6,8,10, SMT siblings 1,3,5,7,9,11;
E-cores 12-15 are never included.

| `THREADS` | `taskset` set | Meaning |
|---|---|---|
| 12 | `0-11` | all P-core logical CPUs, SMT included |
| 8 | `0,1,2,3,4,6,8,10` | 6 primaries + first 2 SMT siblings |
| 6 | `0,2,4,6,8,10` | one logical thread per physical P-core, SMT idle |
| 4 | `0,2,4,6` | first 4 primaries |
| 2 | `0,2` | first 2 primaries |
| 1 | `0` | first primary (scaling MT runs use `ST_CPU` instead) |

General rule: the first `THREADS` primaries, then SMT siblings
(1,3,5,…) to fill, printed numerically sorted (`0-11` for 12).

## Scaling curve

`bash bench/scaling.sh` measures the multi-thread variants at
`THREADS ∈ {1, 2, 4, 6, 8, 12}` (override with `SCALING_THREADS="…"`) and
writes `results/scaling.csv`:

`language,example,variant,threads,n,ms,speedup_vs_1`

with `speedup_vs_1 = ms(threads=1) / ms(threads)` per
`(language,example,variant)`. It runs each parallel-capable binary once per
`THREADS` value (C++ `mandelbrot`; Rust `rayon` × 4; Go `goroutines` × 3;
Python `multiprocessing` × 3), keeps only the multi-thread-variant rows, and
requires the binaries to already exist (it never rebuilds the language
folders). `n = 4000000` for `vector_add`/`dot_product`/`sum_reduce`;
`mandelbrot` uses each lane's default. Pinning: one P-core for `THREADS=1`,
the derived set otherwise. Python `multiprocessing` has no `THREADS=1` row by
lane design (`bench.py` skips a 1-worker pool), so its scaling rows start at
`THREADS=2` with an empty `speedup_vs_1`.

## Files

| Path | Role |
|---|---|
| `run_all.sh` | Build + run + capture driver |
| `scaling.sh` | Thread-scaling curve driver |
| `affinity.sh` | THREADS → P-core `taskset` set mapping (sourced) |
| `aggregate.py` | Merge, correctness gate, CSV and markdown renderer |
| `../results/raw/` | Captured `BENCH,` lines, stderr logs, build logs, `build_status.txt`, skips |
| `../results/results.csv` | Merged rows with correctness status |
| `../results/raw_scaling/` | Per-THREADS scaling captures and logs |
| `../results/scaling.csv` | Scaling curve (`speedup_vs_1` per variant) |
| `../docs/RESULTS.md` | Generated report |

## Noise

The governor is `powersave` with turbo on, so pinned medians vary ~10-15%.
Comparison and tie decisions use the reported **median**; `min` is an advisory
stability diagnostic printed to stderr and is not consumed for ranking
(contract v2.1 measurement rule 7). `RESULTS.md` flags differences under ~15%
as ties and never narrates an ordering inside that band as an effect.
