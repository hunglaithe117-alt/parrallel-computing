//! Ring-commit MPSC demo (Gate 1): producer writes buf[i]=v, then bumps head;
//! consumer loads head, then reads the slot.
//! NOTE: Ring is an N-slot batch-publish buffer that never wraps (not a full
//! ring); it exists to demo commit-vs-payload ordering, nothing more.
//! Broken: head Relaxed (commit visible before payload) -> stale read on weak memory.
//! Fixed: head.store(Release) + head.load(Acquire) -> happens-before.
//! Reference: kern commit 2ed147f (release/acquire pairing for ring commit).

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const N: usize = 8;

struct Ring {
    buf: [AtomicUsize; N],
    head: AtomicUsize,
}

impl Ring {
    fn new() -> Self {
        Self {
            buf: std::array::from_fn(|_| AtomicUsize::new(0)),
            head: AtomicUsize::new(0),
        }
    }
    fn produce(&self, broken: bool) {
        let commit = if broken { Ordering::Relaxed } else { Ordering::Release };
        for i in 0..N {
            self.buf[i].store(i + 1, Ordering::Relaxed); // payload first
            self.head.store(i + 1, commit); // commit: must publish AFTER payload
        }
    }
    fn consume(&self, broken: bool) -> usize {
        let load = if broken { Ordering::Relaxed } else { Ordering::Acquire };
        while self.head.load(load) < N {
            std::hint::spin_loop();
        }
        (0..N).filter(|&i| self.buf[i].load(Ordering::Relaxed) != i + 1).count()
    }
}

fn stress(broken: bool, iters: usize) -> usize {
    let mut bad = 0;
    for _ in 0..iters {
        let ring = Arc::new(Ring::new());
        let p = Arc::clone(&ring);
        let t = std::thread::spawn(move || p.produce(broken));
        bad += ring.consume(broken);
        t.join().unwrap();
    }
    bad
}

fn main() {
    let iters = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2000);
    println!("x86-64 native stress (NEGATIVE CONTROL for StoreStore visibility):");
    println!("  broken (Relaxed head): {} mismatches / {} iters", stress(true, iters), iters);
    println!("  fixed (Release/Acquire): {} mismatches / {} iters", stress(false, iters), iters);
    println!("  TSO preserves StoreStore order, so broken passes here; see `cargo test` (loom) for the real failure.");
}

#[cfg(test)]
mod loom_tests {
    use super::N;
    use loom::sync::{atomic::AtomicUsize as LAtomic, Arc as LArc};
    use std::sync::atomic::Ordering;

    fn model(broken: bool) {
        // Sync logic mirrors Ring::produce/consume above (payload-then-commit,
        // spin-on-head, check slots); loom just replays it under every schedule.
        loom::model(move || {
            let head = LArc::new(LAtomic::new(0));
            let buf: LArc<[LAtomic; N]> = LArc::new(std::array::from_fn(|_| LAtomic::new(0)));
            let (h2, b2) = (head.clone(), buf.clone());
            let prod = loom::thread::spawn(move || {
                for i in 0..N {
                    b2[i].store(i + 1, Ordering::Relaxed);
                    h2.store(i + 1, if broken { Ordering::Relaxed } else { Ordering::Release });
                }
            });
            // Single commit read: loom explores every visibility interleaving.
            let h = head.load(if broken { Ordering::Relaxed } else { Ordering::Acquire });
            for i in 0..h {
                assert_eq!(buf[i].load(Ordering::Relaxed), i + 1, "stale payload after head commit");
            }
            prod.join().unwrap();
        });
    }

    #[test]
    fn fixed_passes() {
        model(false);
    }

    #[test]
    #[should_panic(expected = "stale payload")]
    fn broken_fails() {
        model(true);
    }
}
