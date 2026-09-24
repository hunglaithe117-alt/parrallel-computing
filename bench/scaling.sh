#!/usr/bin/env bash
#
# Thread-scaling curve for the multi-thread variants.
#
# Runs the parallel-capable binaries once per THREADS value in
# {1, 2, 4, 6, 8, 12} and writes results/scaling.csv with columns
#   language,example,variant,threads,n,ms,speedup_vs_1
# where speedup_vs_1 = ms(threads=1) / ms(threads) per (language,example,variant).
#
# Binaries measured (multi-thread variants only; other emitted rows are ignored):
#   C++    mandelbrot                                  (openmp)
#   Rust   vector_add dot_product sum_reduce mandelbrot (rayon)
#   Go     vector_add dot_product sum_reduce           (goroutines)
#   Python vector_add dot_product sum_reduce           (multiprocessing)
#
# n = 4,000,000 for vector_add/dot_product/sum_reduce; mandelbrot keeps each
# lane's default size (it is excluded from cross-language ranking anyway).
# Pinning follows docs/BENCH_CONTRACT.md (v2.1): one P-core for THREADS=1
# (taskset -c 2), the THREADS-derived P-core set otherwise (see
# bench/affinity.sh). Python at THREADS=1 runs pinned to one physical P-core
# (taskset -c 2,3); bench.py skips the multiprocessing variant when the pool
# would have a single worker, so python multiprocessing rows start at
# THREADS=2 and carry an empty speedup_vs_1.
#
# Usage:
#   bash bench/scaling.sh
#
# Environment overrides:
#   SCALING_THREADS="1 2 4 6 8 12"   THREADS values to measure
#   REPS=7            timed repetitions forwarded via BENCH_REPS
#   ST_CPU=2          CPU for the THREADS=1 runs
#   PYTHON_BIN=python3  interpreter used for the CSV computation
#
# Binaries must already exist (run bench/run_all.sh first); this script only
# reads them, never rebuilds the language folders.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$LAB_ROOT"

# shellcheck source=affinity.sh
source "$SCRIPT_DIR/affinity.sh"

THREADS_LIST="${SCALING_THREADS:-1 2 4 6 8 12}"
REPS="${REPS:-7}"
ST_CPU="${ST_CPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
N_BIG=4000000

SCAL_DIR="$LAB_ROOT/results/raw_scaling"
SCAL_CSV="$LAB_ROOT/results/scaling.csv"
mkdir -p "$SCAL_DIR/logs"

RUST_SCALING_EXAMPLES=(vector_add dot_product sum_reduce mandelbrot)
GO_SCALING_EXAMPLES=(vector_add dot_product sum_reduce)
PY_SCALING_EXAMPLES=(vector_add dot_product sum_reduce)

# --- binary presence (read-only; never rebuild here) ---------------------------
[[ -x cpp/bin/mandelbrot ]] || {
    echo "scaling: missing cpp/bin/mandelbrot (run bench/run_all.sh first)" >&2
    exit 1
}
for ex in "${RUST_SCALING_EXAMPLES[@]}"; do
    [[ -x rust/target/release/$ex ]] || {
        echo "scaling: missing rust/target/release/$ex (run bench/run_all.sh first)" >&2
        exit 1
    }
done
for ex in "${GO_SCALING_EXAMPLES[@]}"; do
    [[ -x go/bin/$ex ]] || {
        echo "scaling: missing go/bin/$ex (run bench/run_all.sh first)" >&2
        exit 1
    }
done
[[ -x python/.venv/bin/python ]] || {
    echo "scaling: missing python/.venv (run bench/run_all.sh first)" >&2
    exit 1
}

# Recreate the capture set so the run is idempotent.
# shellcheck disable=SC2086
for lang in cpp rust go python; do
    for t in $THREADS_LIST; do
        : >"$SCAL_DIR/raw_${lang}_t${t}.txt"
    done
done

echo "scaling: lab_root=$LAB_ROOT threads=[$THREADS_LIST] reps=$REPS" >&2

run_mt() {
    local lang="$1" threads="$2" pin="$3" example="$4" n="${5:-}"
    local raw="$SCAL_DIR/raw_${lang}_t${threads}.txt"
    local log="$SCAL_DIR/logs/${lang}_t${threads}_${example}.log"

    local -a cmd=()
    local -a envs=(BENCH_REPS="$REPS" THREADS="$threads"
        OMP_NUM_THREADS="$threads" RAYON_NUM_THREADS="$threads"
        GOMAXPROCS="$threads")
    case "$lang" in
        cpp)
            cmd=(cpp/bin/"$example")
            ;;
        rust)
            cmd=(rust/target/release/"$example")
            [[ -n "$n" ]] && envs+=(BENCH_N="$n")
            ;;
        go)
            cmd=(go/bin/"$example")
            [[ -n "$n" ]] && envs+=(BENCH_N="$n")
            ;;
        python)
            cmd=(python/.venv/bin/python python/bench.py "$example")
            [[ -n "$n" ]] && cmd+=(--n "$n")
            ;;
    esac

    local tmp="$SCAL_DIR/.tmp_${lang}_t${threads}_${example}.out"
    local rc=0
    env "${envs[@]}" taskset -c "$pin" "${cmd[@]}" >"$tmp" 2>"$log" || rc=$?
    if grep -q '^BENCH,' "$tmp"; then
        grep '^BENCH,' "$tmp" >>"$raw"
    else
        echo "scaling: WARNING: no BENCH, lines $lang/$example threads=$threads (exit $rc; see $log)" >&2
    fi
    rm -f "$tmp"
}

# shellcheck disable=SC2086
for t in $THREADS_LIST; do
    if ((t == 1)); then
        pin="$ST_CPU"
    else
        pin="$(pcore_set "$t")"
    fi
    echo "scaling: threads=$t pin=$pin n(big)=$N_BIG" >&2

    run_mt cpp "$t" "$pin" mandelbrot

    for ex in "${RUST_SCALING_EXAMPLES[@]}"; do
        if [[ "$ex" == mandelbrot ]]; then
            run_mt rust "$t" "$pin" "$ex"
        else
            run_mt rust "$t" "$pin" "$ex" "$N_BIG"
        fi
    done

    for ex in "${GO_SCALING_EXAMPLES[@]}"; do
        run_mt go "$t" "$pin" "$ex" "$N_BIG"
    done

    for ex in "${PY_SCALING_EXAMPLES[@]}"; do
        if ((t == 1)); then
            # One physical P-core (both SMT siblings): bench.py skips the
            # multiprocessing variant for a 1-worker pool, so this pass only
            # documents the skip; python scaling rows start at THREADS=2.
            run_mt python "$t" "2,3" "$ex" "$N_BIG"
        else
            run_mt python "$t" "$pin" "$ex" "$N_BIG"
        fi
    done
done

# --- scaling.csv ---------------------------------------------------------------
# Keep multi-thread variants only; speedup_vs_1 is the threads=1 median over
# the same (language,example,variant). Python multiprocessing has no threads=1
# row by lane design, so its speedup_vs_1 stays empty (see header above).
echo "scaling: writing $SCAL_CSV" >&2
"$PYTHON_BIN" - "$SCAL_DIR" "$SCAL_CSV" "$THREADS_LIST" <<'EOF'
import csv
import sys
from pathlib import Path

scal_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
threads_list = [int(t) for t in sys.argv[3].split()]

LANGS = ["cpp", "rust", "go", "python"]
MT_VARIANTS = {"openmp", "pthreads", "std_threads", "rayon", "goroutines",
               "multiprocessing"}
EX_ORDER = {"vector_add": 0, "dot_product": 1, "sum_reduce": 2, "mandelbrot": 3}

data: dict[tuple, tuple[int, float]] = {}
for lang in LANGS:
    for t in threads_list:
        path = scal_dir / f"raw_{lang}_t{t}.txt"
        if not path.exists():
            print(f"scaling: WARNING missing capture {path}", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            if not line.startswith("BENCH,"):
                continue
            fields = line.split(",")
            if len(fields) != 6:
                continue
            _, example, variant, raw_n, raw_ms, _checksum = fields
            if variant not in MT_VARIANTS:
                continue
            data.setdefault((lang, example, variant, t),
                            (int(raw_n), float(raw_ms)))

rows = []
missing_baselines = set()
for (lang, example, variant, t), (n, ms) in data.items():
    baseline = data.get((lang, example, variant, threads_list[0]))
    if baseline is None:
        missing_baselines.add((lang, example, variant))
        speedup = ""
    else:
        speedup = f"{baseline[1] / ms:.2f}"
    rows.append((LANGS.index(lang), EX_ORDER.get(example, 99), variant, t,
                 lang, example, n, ms, speedup))
rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["language", "example", "variant", "threads", "n", "ms",
                     "speedup_vs_1"])
    for _, _, variant, t, lang, example, n, ms, speedup in rows:
        writer.writerow([lang, example, variant, t, n, f"{ms:.4f}", speedup])
for lang, example, variant in sorted(missing_baselines):
    print(f"scaling: WARNING no threads={threads_list[0]} baseline for "
          f"{lang}/{example}/{variant}; speedup left empty", file=sys.stderr)
print(f"scaling: {len(rows)} row(s); "
      f"{len(missing_baselines)} key(s) without baseline", file=sys.stderr)
EOF

echo "scaling: done ($SCAL_CSV)" >&2
