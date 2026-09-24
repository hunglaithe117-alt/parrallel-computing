# Phase 2 (Rust): ring-commit ordering — `rust-ring`

## Concept

MPSC ring commit: the producer writes the payload first (`buf[i] = v`), then
bumps `head`; the consumer loads `head`, then reads the slot. The commit must
publish *after* the payload, or the consumer can observe the new `head` while
reading a stale slot. The fix is a release/acquire pair:
`head.store(Release)` synchronizes-with `head.load(Acquire)`, giving the
consumer happens-before over every payload write before the commit.
Reference: kern commit `2ed147f`.

## What breaks

Broken mode publishes the commit with `Ordering::Relaxed`. On a weak memory
model the head store may become visible before the payload store, so the
consumer sees `head == i+1` yet reads `buf[i] == 0`. Loom reproduces this:
`loom_tests::broken_fails` panics with `stale payload after head commit`.

## Fix

One-line-class change, same code shape: commit with `Release`, observe with
`Acquire` (`loom_tests::fixed_passes`). No fences, no seqcst, no extra copies.

## How to run

From the project root (Rust 1.96.1 stable; loom 0.7 is a dev-dependency):

```bash
cargo build --manifest-path sync-learn/rust-ring/Cargo.toml
cargo test --manifest-path sync-learn/rust-ring/Cargo.toml
cargo run --manifest-path sync-learn/rust-ring/Cargo.toml -- [iters]
```

## Expected output

`cargo test` (loom model: broken panics as required, fixed passes):

```text
running 2 tests
test loom_tests::broken_fails - should panic ... ok
test loom_tests::fixed_passes ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

`cargo run` (x86-64 native stress — NEGATIVE CONTROL for StoreStore
visibility; TSO preserves StoreStore order, so the broken mode cannot fail
here and the loom pair above is the real evidence):

```text
x86-64 native stress (NEGATIVE CONTROL for StoreStore visibility):
  broken (Relaxed head): 0 mismatches / 2000 iters
  fixed (Release/Acquire): 0 mismatches / 2000 iters
  TSO preserves StoreStore order, so broken passes here; see `cargo test` (loom) for the real failure.
```

Demo is 105 lines total (`src/main.rs` + `Cargo.toml`); `cargo clippy --all-targets`
is clean.
