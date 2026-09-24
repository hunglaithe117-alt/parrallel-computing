// storebuffer-visibility: message passing x=1;ready=true.
// BROKEN: plain x/ready (data race, needs release/acquire).
// FIXED: atomic release/acquire. x86 run is NEGATIVE CONTROL only
// (TSO hides the bug; litmus.txt shows the forbidden outcome).
// Build: g++ -O2 -std=c++17 demo.cpp -o demo[_broken] [-DBROKEN]
#include <atomic>
#include <cstdio>
#include <thread>
#ifdef BROKEN
int x = 0;
bool ready = false;
#else
std::atomic<int> x{0};
std::atomic<bool> ready{false};
#endif
int main() {
    int bad = 0;
    for (int it = 0; it < 200000; ++it) {
#ifdef BROKEN
        x = 0; ready = false;
#else
        x.store(0, std::memory_order_relaxed);
        ready.store(false, std::memory_order_relaxed);
#endif
        int seen = -1;
        std::thread w([] {
#ifdef BROKEN
            x = 1; ready = true;  // BUG: no ordering, data race
#else
            x.store(1, std::memory_order_relaxed);
            ready.store(true, std::memory_order_release);
#endif
        });
        std::thread r([&] {
#ifdef BROKEN
            while (!ready) {}  // BUG: may never see it / see x==0
            seen = x;
#else
            while (!ready.load(std::memory_order_acquire)) {}
            seen = x.load(std::memory_order_relaxed);
#endif
        });
        w.join(); r.join();
        if (seen != 1) ++bad;
    }
#ifdef BROKEN
    std::printf("[x86] BROKEN garbage=%d/200000 (compiler reordered plain stores)\n", bad);
#else
    std::printf("[x86] FIXED garbage=%d/200000 (0 here shows the compiler fix; forbidden per litmus/C++ model)\n", bad);
#endif
    return 0;
}
