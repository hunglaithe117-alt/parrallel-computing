# Phase 3 C++ lane: lock handover + sharded counters

Source: `sync-learn/cpp-locks/` (`tas_vs_ticket_vs_mcs.cpp` 87 lines,
`sharded_counter.cpp` 45 lines). Recorded on i5-14400, `taskset -c 0-11`.

## 1. TAS vs ticket vs MCS handover
- Concept: under contention the handover path is the cost; TAS spins on one
  global line (can starve), ticket adds FIFO fairness, MCS queues waiters so
  each spins on its own line (cf. kernel qspinlock: TAS fast-path, MCS
  slow-path).
- What collapses: single-flag spinning serializes on one cache line; ticket
  pays extra for fairness (265.8ms vs TAS 146.5ms here).
- Fix: MCS-style local-spin queue (160.8ms here — local-spin by
  construction; ms parity here, traffic win needs perf/pinning).
- Run: `g++ -O2 -std=c++17 -pthread tas_vs_ticket_vs_mcs.cpp -o locks && ./locks 8 200000`
- Expected: `TAS 146.5ms OK / TICKET 265.8ms OK (FIFO-fair) / MCS 160.8ms OK`
  (fairness by construction, not measured: counter==expect proves exclusion
  only, no starvation histogram).

## 2. Sharded counter
- Concept: one shared atomic collapses as threads grow; per-thread
  (`alignas(64)`) lanes scale ~linearly, repaid by one O(T) sum (per-CPU lesson).
- What collapses: 8 threads, single-atomic 246.6ms; sharded 22.4ms (11.0x).
  At 2 threads the gap is only 1.3x — contention, not work, is the cost.
- Fix: shard updates, sum once at the end; all sums exact.
- Run: `g++ -O2 -std=c++17 -pthread sharded_counter.cpp -o sharded && ./sharded 8 2000000`
- Expected: `single-atomic 246.6ms OK / sharded 22.4ms OK (11.0x)`

## Perf note (optional)
`perf stat -e cache-misses,cycles ./sharded 8 2000000` is the right probe for
the collapse (misses concentrate on the single line), but this host blocks it
(`perf_event_paranoid=4`); no sample recorded — marked optional.
