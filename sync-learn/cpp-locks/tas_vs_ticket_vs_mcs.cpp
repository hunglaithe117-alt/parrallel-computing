// TAS vs ticket vs MCS-style lock handover (contended increment hammer).
// Ticket = FIFO fairness (TAS can starve latecomers; ticket serves in order).
// MCS-style = MCS-queue sketch only, no TAS fast-path (unlike kernel qspinlock,
// which pairs a TAS fast-path with an MCS slow-path); waiters spin on their own
// cache line instead of hammering one flag. (Sketch: one qnode per thread,
// non-reentrant -- enough to show local-spin handover.)
// Build: g++ -O2 -std=c++17 -pthread tas_vs_ticket_vs_mcs.cpp -o locks
// Run: ./locks [threads=8] [iters/thread=200000]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

struct TASLock {
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    void lock() { while (f.test_and_set(std::memory_order_acquire)) {} }
    void unlock() { f.clear(std::memory_order_release); }  // global spin line
};

struct TicketLock {
    std::atomic<unsigned> next{0}, now{0};
    void lock() {  // FIFO: my turn == ticket; fairness by construction
        unsigned t = next.fetch_add(1, std::memory_order_relaxed);
        while (now.load(std::memory_order_acquire) != t) {}
    }
    void unlock() { now.fetch_add(1, std::memory_order_release); }
};

struct MCSLock {
    struct QNode { std::atomic<QNode*> next{nullptr}; std::atomic<bool> locked{false}; };
    std::atomic<QNode*> tail{nullptr};
    void lock(QNode& q) {
        q.next.store(nullptr, std::memory_order_relaxed);
        QNode* pred = tail.exchange(&q, std::memory_order_acquire);
        if (!pred) return;
        q.locked.store(true, std::memory_order_relaxed);
        pred->next.store(&q, std::memory_order_release);
        while (q.locked.load(std::memory_order_acquire)) {}  // LOCAL spin only
    }
    void unlock(QNode& q) {
        QNode* succ = q.next.load(std::memory_order_acquire);
        if (!succ) {
            QNode* e = &q;
            if (tail.compare_exchange_strong(e, nullptr, std::memory_order_release)) return;
            while (!(succ = q.next.load(std::memory_order_acquire))) {}
        }
        succ->locked.store(false, std::memory_order_release);  // handover
    }
};

template <class F>
static double hammer(int nt, long iters, F&& op) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, iters] { for (long i = 0; i < iters; ++i) op(); });
    for (auto& th : ts) th.join();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int nt = argc > 1 ? std::atoi(argv[1]) : 8;
    long iters = argc > 2 ? std::atol(argv[2]) : 200000L;
    long counter = 0;
    TASLock tas;
    double t_tas = hammer(nt, iters, [&] { tas.lock(); ++counter; tas.unlock(); });
    long expect = (long)nt * iters;
    std::printf("TAS     %8.1f ms  counter=%ld/%ld %s\n", t_tas, counter, expect,
                counter == expect ? "OK" : "LOST!");
    counter = 0;
    TicketLock ticket;
    double t_tick = hammer(nt, iters, [&] { ticket.lock(); ++counter; ticket.unlock(); });
    std::printf("TICKET  %8.1f ms  counter=%ld/%ld %s (FIFO-fair)\n", t_tick, counter,
                expect, counter == expect ? "OK" : "LOST!");
    counter = 0;
    MCSLock mcs;
    // One qnode per thread, built once outside the iteration loop (not per op).
    auto t0_mcs = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < nt; ++t)
            ts.emplace_back([&, iters] {
                MCSLock::QNode q;
                for (long i = 0; i < iters; ++i) { mcs.lock(q); ++counter; mcs.unlock(q); }
            });
        for (auto& th : ts) th.join();
    }
    double t_mcs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0_mcs).count();
    std::printf("MCS     %8.1f ms  counter=%ld/%ld %s (local-spin handover)\n", t_mcs,
                counter, expect, counter == expect ? "OK" : "LOST!");
    return (counter == expect) ? 0 : 1;
}
