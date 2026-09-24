// compiler-barrier-merged: hoist myth + volatile myth, one file.
// BROKEN (-DBROKEN): plain bool flag; -O2 hoists the load out of
//   while(!flag){} so the spinner may never observe the store (bounded here
//   so the demo terminates: hitting the cap == HOISTED). volatile alone still
//   reloads but is NOT synchronization (data race, no ordering) -- myth.
// FIXED: atomic acquire load (+ explicit kernel-style barrier()); acquire alone
// already blocks the compiler hoist -- barrier() is kept as the compiler-barrier
// demo (observable behavior identical without it). See asm-note.txt.
// Build: g++ -O2 -std=c++17 demo.cpp -o demo[_broken] [-DBROKEN]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#ifdef BROKEN
bool flag = false;  // BUG: hoisted at -O2; volatile would still race
#else
std::atomic<bool> flag{false};
#define barrier() asm volatile("" ::: "memory")
#endif
int main() {
    long spins = 0;
#ifdef BROKEN
    const long CAP = 300000000L;
    flag = false;
#else
    flag.store(false, std::memory_order_relaxed);
#endif
    std::thread w([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
#ifdef BROKEN
        flag = true;
#else
        flag.store(true, std::memory_order_release);
#endif
    });
    auto t0 = std::chrono::steady_clock::now();
#ifdef BROKEN
    while (!flag && ++spins < CAP) {}  // cap proves hoist if hit
#else
    while (!flag.load(std::memory_order_acquire)) { barrier(); ++spins; }
#endif
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    w.join();
#ifdef BROKEN
    std::printf("[positive control x86: hoist observable] BROKEN spins=%ld ms=%lld %s\n", spins,
        (long long)ms, spins >= CAP ? "HOISTED-at-O2" : "observed(volatile-like luck)");
#else
    std::printf("FIXED spins=%ld ms=%lld observed via acquire+barrier\n", spins,
        (long long)ms);
#endif
    return 0;
}
