# C++ lock + sharding demos (phase 3)

| demo | build | run |
|---|---|---|
| TAS vs ticket vs MCS handover | `g++ -O2 -std=c++17 -pthread tas_vs_ticket_vs_mcs.cpp -o locks` | `./locks [threads] [iters]` |
| sharded counter | `g++ -O2 -std=c++17 -pthread sharded_counter.cpp -o sharded` | `./sharded [threads] [iters]` |

Or `make -C sync-learn/cpp-locks run`. Optional: `perf stat -e
cache-misses,cycles ./sharded 8 2000000` (needs perf rights; else skip).
