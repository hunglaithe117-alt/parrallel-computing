// invqueue-loadload: reader does two loads (flag, then payload).
// BROKEN: relaxed flag load lets the payload read go stale/go before it.
// FIXED: acquire flag load orders the second load after it (+ rmb on Linux).
// x86 run is NEGATIVE CONTROL only (TSO orders load-load already).
// Build: g++ -O2 -std=c++17 demo.cpp -o demo[_broken] [-DBROKEN]
#include <atomic>
#include <cstdio>
#include <thread>
std::atomic<int> data{0};
std::atomic<int> flag{0};
int main() {
    int bad = 0;
    for (int it = 0; it < 200000; ++it) {
        data.store(0, std::memory_order_relaxed);
        flag.store(0, std::memory_order_relaxed);
        int got = -1;
        std::thread w([] {
            data.store(42, std::memory_order_relaxed);
            flag.store(1, std::memory_order_release);
        });
        std::thread r([&] {
#ifdef BROKEN
            while (!flag.load(std::memory_order_relaxed)) {}  // BUG: no order
            got = data.load(std::memory_order_relaxed);       // may read stale 0
#else
            while (!flag.load(std::memory_order_acquire)) {}  // rmb/acquire fix
            got = data.load(std::memory_order_relaxed);       // ordered after flag
#endif
        });
        w.join(); r.join();
        if (got != 42) ++bad;
    }
#ifdef BROKEN
    std::printf("[negative-control x86] BROKEN stale=%d/200000 (TSO hides it)\n", bad);
#else
    std::printf("[x86] FIXED stale=%d/200000 (0 here shows the compiler fix; forbidden per litmus/C++ model)\n", bad);
#endif
    return 0;
}
