"""Sleep-rules demo: never sleep while holding a lock.

Kernel mapping (from the article): process context may sleep (schedule) but
must not hold a spinlock across the sleep; irq/atomic context must never
sleep (stated, not shown -- no Python irq equivalent); on SMP sleep orders
nothing -- use lock/barrier for visibility.
Per-CPU pointer: shard contended counters per CPU to avoid the lock entirely.
GIL caveat: GOOD looks parallel only because sleep releases GIL;
CPU-bound work under threads would not parallelize.
"""
import threading
import time

HOLD = 0.2
N = 4


def fanout(target):
    """Run target on N threads, return the start-to-join wall time."""
    ts = [threading.Thread(target=target) for _ in range(N)]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.monotonic() - t0


def bad():
    mu = threading.Lock()

    def w():
        with mu:
            time.sleep(HOLD)  # BAD: sleep inside critical section serializes all

    return fanout(w)


def good():
    mu = threading.Lock()

    def w():
        time.sleep(HOLD)  # GOOD: sleep outside, lock only for the update
        with mu:
            pass

    return fanout(w)


if __name__ == "__main__":
    tb, tg = bad(), good()
    print(f"BAD (sleep-while-holding):  {tb:.2f}s for {N} threads (~serial)")
    print(f"GOOD (sleep-outside-lock):  {tg:.2f}s for {N} threads (~parallel)")
    print("RULE: sleep outside the critical section; irq/atomic ctx never sleeps.")
