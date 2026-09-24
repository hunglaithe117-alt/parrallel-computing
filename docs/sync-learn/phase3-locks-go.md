# Phase 3 Go lane: lock ordering + read-scale demos (Gate 1 evidence)

Source: `sync-learn/go-locks/<demo>/main.go` (60/79 lines) and
`sync-learn/python-sleep/sleep-rules.py` (48 lines). Run Go demos in file
mode (`go run main.go`) — no `go.mod` covers `sync-learn/`, stdlib only.

## 1. lock-order-deadlock
- Concept: two locks need one global acquisition order; opposite orders can
  deadlock (g1 holds A wants B while g2 holds B wants A).
- What breaks: default mode runs A→B vs B→A with a 100ms hold window; the
  1s watchdog fires: `DEADLOCK: g1 holds A wants B, g2 holds B wants A`.
- Fix: `fixed` mode imposes global order A→B in both goroutines; both finish.
- Run: `go run main.go [deadlock|fixed]` in `sync-learn/go-locks/lock-order-deadlock/`
- Expected: deadlock mode prints the `DEADLOCK` line and returns (~1s);
  fixed prints `g1/g2 done` + `FIXED: both goroutines completed under global order A->B`.

## 2. rcu-inspired-readscale
- Concept: rcu-INSPIRED (no grace period — GC + COW), not true RCU. Readers
  `atomic.Pointer[config].Load()` with no lock; the writer swaps a fresh copy
  per update and GC reclaims the old one once unreferenced.
- What breaks: RWMutex baseline serializes readers against the writer
  (`RLock`/`Lock` exclusion, ~45ms wall clock here).
- Fix: copy-on-write swap makes reads lock-free (~0.25ms here); writer cost
  moves to copy+Store. Loads feed a printed sink so DCE cannot drop them.
  Metric is wall-clock throughput (avg ns/read); no per-read latency
  histogram, so no p99 is reported.
- Epoch note: true userspace RCU needs grace periods (crossbeam-epoch/EBR).
- Run: `go run main.go` in `sync-learn/go-locks/rcu-inspired-readscale/`
- Expected: `RWMutex: 1600000 reads in ~45ms (avg ~28ns/read)`,
  `COW: 1600000 reads in ~0.25ms (avg ~0.2ns/read)` plus a nonzero `sink`
  line (ratios host-dependent).

## 3. sleep-rules (Python)
- Concept: never sleep while holding a lock; sleep orders nothing on SMP.
- Mapping (article): process context may sleep but must not hold a spinlock
  across it; irq/atomic context must never sleep; SMP needs lock/barrier for
  visibility, not sleep. Per-CPU pointer: shard hot counters per CPU to skip
  the lock entirely.
- What breaks: `BAD` holds `Lock` across `sleep(0.2)` × 4 threads → ~0.80s
  (serial); `GOOD` sleeps outside, locks only for the update → ~0.20s.
- Run: `python3 sync-learn/python-sleep/sleep-rules.py`
- Expected: `BAD ... ~0.80s (~serial)`, `GOOD ... ~0.20s (~parallel)`, then
  the `RULE:` line. Validated: `go vet` clean on both Go demos.
