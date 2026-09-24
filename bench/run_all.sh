#!/usr/bin/env bash
#
# Cross-language benchmark aggregator for the parallel-computing lab.
#
# Builds all four language lanes, runs every benchmark pinned twice (a
# single-thread pass on one P-core and a multi-thread pass on the whole P-core
# set), captures only stdout `BENCH,` lines, then hands the captures to
# `bench/aggregate.py`, which merges by contract v2.1 precedence, checks
# within-language correctness, writes `results/results.csv`, and regenerates
# `docs/RESULTS.md`.
#
# Usage:
#   bash bench/run_all.sh            # full run (n = 4,000,000 per cross-language kernel)
#   bash bench/run_all.sh --quick    # small-n / fewer-rep smoke run
#
# Environment overrides:
#   THREADS=12        multi-thread pool size (also OMP/RAYON/GOMAXPROCS);
#                     the multi-thread CPU set derives from it (see bench/affinity.sh)
#   P_SET=            (optional) explicit taskset CPU set for the multi-thread
#                     pass; default derives from THREADS (12 -> 0-11,
#                     6 -> 0,2,4,6,8,10)
#   ST_CPU=2          CPU for the single-thread pass
#   REPS=7            timed repetitions forwarded via BENCH_REPS
#   RUN_TAG=          (optional) tag suffix for outputs: raw captures go to
#                     results/raw_<tag>/, csv to results/results_<tag>.csv,
#                     report to docs/RESULTS_<tag>.md; empty keeps defaults
#   PYTHON_BIN=python3  interpreter used to run bench/aggregate.py
#
# See bench/README.md and docs/BENCH_CONTRACT.md (v2.1).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$LAB_ROOT"

# shellcheck source=affinity.sh
source "$SCRIPT_DIR/affinity.sh"

THREADS="${THREADS:-12}"
if [[ -z "${P_SET:-}" ]]; then
    P_SET="$(pcore_set "$THREADS")"
fi
ST_CPU="${ST_CPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-}"
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "run_all: RUN_TAG must match ^[A-Za-z0-9_-]+\$, got '$RUN_TAG'" >&2
    exit 2
fi

QUICK=0

usage() {
    cat >&2 <<'EOF'
usage: bash bench/run_all.sh [--quick]

  --quick   smoke mode: n = 200000 for the cross-language kernels and 3 reps
EOF
}

while (($#)); do
    case "$1" in
        --quick) QUICK=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "run_all: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

REPS_DEFAULT=7
if ((QUICK)); then
    REPS_DEFAULT=3
fi
REPS="${REPS:-$REPS_DEFAULT}"

# Cross-language element counts. `--quick` shrinks them for smoke testing;
# mandelbrot and false_sharing_* keep each lane's own default (they are excluded
# from cross-language ranking anyway, see docs/BENCH_CONTRACT.md v2.1 N2).
if ((QUICK)); then
    N_VECTOR_ADD=200000
    N_SAXPY=200000
    N_DOT_PRODUCT=200000
    N_SUM_REDUCE=200000
else
    N_VECTOR_ADD=4000000
    N_SAXPY=4000000
    N_DOT_PRODUCT=4000000
    N_SUM_REDUCE=4000000
fi

if [[ -n "$RUN_TAG" ]]; then
    RAW_DIR="$LAB_ROOT/results/raw_${RUN_TAG}"
    RESULTS_CSV="$LAB_ROOT/results/results_${RUN_TAG}.csv"
    RESULTS_MD="$LAB_ROOT/docs/RESULTS_${RUN_TAG}.md"
else
    RAW_DIR="$LAB_ROOT/results/raw"
    RESULTS_CSV="$LAB_ROOT/results/results.csv"
    RESULTS_MD="$LAB_ROOT/docs/RESULTS.md"
fi
LOG_DIR="$RAW_DIR/logs"
BUILD_STATUS="$RAW_DIR/build_status.txt"
mkdir -p "$RAW_DIR" "$LOG_DIR"

echo "run_all: lab_root=$LAB_ROOT threads=$THREADS pin(ST)=$ST_CPU pin(MT)=$P_SET reps=$REPS quick=$QUICK tag=${RUN_TAG:-<none>}" >&2

# --- builds ------------------------------------------------------------------
# Every outcome is recorded in $BUILD_STATUS so the report can surface a failed
# lane instead of silently presenting stale binaries as fresh (Gate 3 m4).
build_one() {
    local lang="$1"
    shift
    local log="$RAW_DIR/build_${lang}.log"
    echo "run_all: building $lang: $*" >&2
    if "$@" >"$log" 2>&1; then
        printf 'BUILD,%s,ok\n' "$lang" >>"$BUILD_STATUS"
        echo "run_all: $lang build ok" >&2
        return 0
    fi
    printf 'BUILD,%s,FAIL\n' "$lang" >>"$BUILD_STATUS"
    echo "run_all: WARNING: $lang build failed; see $log (continuing with existing binaries)" >&2
    return 1
}

: >"$BUILD_STATUS"
build_one cpp make -C cpp || true
build_one rust env RUSTFLAGS="-C target-cpu=native" \
    cargo build --release --manifest-path rust/Cargo.toml || true
build_one go make -C go build || true
if [[ -d python/.venv ]]; then
    build_one python make -C python lib || true
else
    build_one python make -C python setup || true
    build_one python make -C python lib || true
fi

# --- captures ----------------------------------------------------------------
# Recreate the capture set so the run is idempotent.
for lang in cpp rust go python; do
    for run_pass in st mt; do
        : >"$RAW_DIR/raw_${lang}_${run_pass}.txt"
    done
done
: >"$RAW_DIR/skips.txt"

CPP_EXAMPLES=(vector_add saxpy dot_product sum_reduce mandelbrot
    false_sharing_adjacent false_sharing_padded)
RUST_EXAMPLES=(vector_add saxpy dot_product sum_reduce mandelbrot)
GO_EXAMPLES=(vector_add dot_product sum_reduce
    false_sharing_adjacent false_sharing_padded)
PYTHON_EXAMPLES=(vector_add dot_product sum_reduce)

n_for() {
    case "$1" in
        vector_add) printf '%s' "$N_VECTOR_ADD" ;;
        saxpy) printf '%s' "$N_SAXPY" ;;
        dot_product) printf '%s' "$N_DOT_PRODUCT" ;;
        sum_reduce) printf '%s' "$N_SUM_REDUCE" ;;
        *) printf '' ;;
    esac
}

record_skip() {
    local lang="$1" example="$2" run_pass="$3" reason="$4"
    printf 'SKIP,%s,%s,%s,%s\n' "$lang" "$example" "$run_pass" "$reason" >>"$RAW_DIR/skips.txt"
    echo "run_all: SKIP $lang/$example ($run_pass): $reason" >&2
}

run_one() {
    local lang="$1" run_pass="$2" example="$3"
    local raw="$RAW_DIR/raw_${lang}_${run_pass}.txt"
    local log="$LOG_DIR/${lang}_${run_pass}_${example}.log"
    local n
    n="$(n_for "$example")"

    local -a cmd=()
    case "$lang" in
        cpp)
            [[ -x "cpp/bin/$example" ]] || {
                record_skip "$lang" "$example" "$run_pass" "binary missing"
                return 0
            }
            cmd=(cpp/bin/"$example")
            [[ -n "$n" ]] && cmd+=(--n "$n")
            ;;
        rust)
            [[ -x "rust/target/release/$example" ]] || {
                record_skip "$lang" "$example" "$run_pass" "binary missing"
                return 0
            }
            cmd=(rust/target/release/"$example")
            ;;
        go)
            [[ -x "go/bin/$example" ]] || {
                record_skip "$lang" "$example" "$run_pass" "binary missing"
                return 0
            }
            cmd=(go/bin/"$example")
            ;;
        python)
            [[ -x "python/.venv/bin/python" ]] || {
                record_skip "$lang" "$example" "$run_pass" "venv missing"
                return 0
            }
            cmd=(python/.venv/bin/python python/bench.py "$example")
            [[ -n "$n" ]] && cmd+=(--n "$n")
            ;;
    esac

    local -a envs=(BENCH_REPS="$REPS")
    local -a pin
    if [[ "$run_pass" == mt ]]; then
        envs+=(THREADS="$THREADS" OMP_NUM_THREADS="$THREADS"
            RAYON_NUM_THREADS="$THREADS" GOMAXPROCS="$THREADS")
        pin=(taskset -c "$P_SET")
    else
        pin=(taskset -c "$ST_CPU")
    fi
    if [[ -n "$n" && ("$lang" == rust || "$lang" == go) ]]; then
        envs+=(BENCH_N="$n")
    fi

    local tmp="$RAW_DIR/.tmp_${lang}_${run_pass}_${example}.out"
    local rc=0
    env "${envs[@]}" "${pin[@]}" "${cmd[@]}" >"$tmp" 2>"$log" || rc=$?

    local captured=0
    if grep -q '^BENCH,' "$tmp"; then
        captured="$(grep -c '^BENCH,' "$tmp")"
        grep '^BENCH,' "$tmp" >>"$raw"
    fi
    rm -f "$tmp"

    if ((captured == 0)); then
        record_skip "$lang" "$example" "$run_pass" "no BENCH, lines (exit $rc)"
    elif ((rc != 0)); then
        echo "run_all: $lang/$example ($run_pass): $captured rows, non-zero exit $rc (kept; see $log)" >&2
    fi
}

for lang in cpp rust go python; do
    case "$lang" in
        cpp) examples=("${CPP_EXAMPLES[@]}") ;;
        rust) examples=("${RUST_EXAMPLES[@]}") ;;
        go) examples=("${GO_EXAMPLES[@]}") ;;
        python) examples=("${PYTHON_EXAMPLES[@]}") ;;
    esac
    for run_pass in st mt; do
        echo "run_all: $lang $run_pass pass" >&2
        for example in "${examples[@]}"; do
            run_one "$lang" "$run_pass" "$example"
        done
    done
done

# --- aggregate ---------------------------------------------------------------
echo "run_all: aggregating" >&2
aggregate_args=(
    --raw-dir "$RAW_DIR"
    --csv "$RESULTS_CSV"
    --md "$RESULTS_MD"
    --threads "$THREADS"
    --reps "$REPS"
    --tag "$RUN_TAG"
    --pin-set "$P_SET"
)
if ((QUICK)); then
    aggregate_args+=(--quick)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/aggregate.py" "${aggregate_args[@]}"

echo "run_all: done ($RESULTS_CSV, $RESULTS_MD)" >&2
