// Sharded counter: single atomic vs per-thread sharding (per-CPU lesson).
// Single fetch_add serializes on one cache line (contention collapse as
// threads grow); sharded lanes scale ~linearly, paid back by one O(T) sum.
// Build: g++ -O2 -std=c++17 -pthread sharded_counter.cpp -o sharded
// Run: ./sharded [threads=8] [iters/thread=2000000]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

struct alignas(64) Lane { std::atomic<long> v{0}; };  // one line per lane

template <class F>
static double run(int nt, F&& op) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) ts.emplace_back(op, t);
    for (auto& th : ts) th.join();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int nt = argc > 1 ? std::atoi(argv[1]) : 8;
    long iters = argc > 2 ? std::atol(argv[2]) : 2000000L;
    long expect = (long)nt * iters;
    std::atomic<long> single{0};  // COLLAPSE: all threads hammer one line
    double t1 = run(nt, [&](int) {
        for (long i = 0; i < iters; ++i) single.fetch_add(1, std::memory_order_relaxed);
    });
    std::printf("single-atomic %8.1f ms  sum=%ld/%ld %s\n", t1, single.load(),
                expect, single.load() == expect ? "OK" : "LOST!");
    std::vector<Lane> lanes(nt);  // FIX: private line per thread, sum once
    double t2 = run(nt, [&](int t) {
        for (long i = 0; i < iters; ++i)
            lanes[t].v.fetch_add(1, std::memory_order_relaxed);
    });
    long sum = 0;
    for (auto& l : lanes) sum += l.v.load(std::memory_order_relaxed);
    std::printf("sharded       %8.1f ms  sum=%ld/%ld %s (%.1fx)\n", t2, sum, expect,
                sum == expect ? "OK" : "LOST!", t1 / t2);
    return (single.load() == expect && sum == expect) ? 0 : 1;
}
