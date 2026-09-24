#!/usr/bin/env bash
# Shared P-core affinity mapping for the bench harness.
#
# On this box (i5-14400) the primary P-core logical CPUs are 0,2,4,6,8,10 and
# their SMT siblings are 1,3,5,7,9,11; E-cores are 12-15 and must never be
# included (docs/BENCH_CONTRACT.md v2.1: P-cores only, physical vs logical
# stated explicitly).
#
# `pcore_set <n>` prints the taskset CPU list for n threads: the first n
# primaries, then SMT siblings to fill. Sourced by run_all.sh and scaling.sh.

# Print the taskset CPU set for n threads (1..12). Exits non-zero on bad input.
pcore_set() {
    local n="${1:-}"
    if ! [[ "$n" =~ ^[0-9]+$ ]] || ((n < 1 || n > 12)); then
        echo "pcore_set: want THREADS in 1..12 (P-core logical CPUs only), got '${1:-}'" >&2
        return 1
    fi
    if ((n == 12)); then
        printf '0-11\n'
        return 0
    fi
    local -a prim=(0 2 4 6 8 10) sib=(1 3 5 7 9 11) out=()
    local i
    for ((i = 0; i < n && i < 6; i++)); do out+=("${prim[$i]}"); done
    for ((i = 0; i < n - 6; i++)); do out+=("${sib[$i]}"); done
    mapfile -t out < <(printf '%s\n' "${out[@]}" | sort -n)
    local IFS=,
    printf '%s\n' "${out[*]}"
}
