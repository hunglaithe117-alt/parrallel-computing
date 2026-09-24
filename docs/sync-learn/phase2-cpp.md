# Phase 2 C++ lane: memory-model demos (Gate 1 evidence)

Source: `sync-learn/cpp/<demo>/demo.cpp` (`-DBROKEN` = broken mode, <80 lines
each); litmus in `litmus.txt`; asm note in `compiler-barrier-merged/asm-note.txt`.
x86 loop output for the HW demos (storebuffer, invqueue) is a **negative
control** (TSO hides HW-only reorderings); the compiler demo's broken mode is
a **[positive control x86: hoist observable]**.

## 1. storebuffer-visibility
- Concept: message passing needs release/acquire, not just store order.
- What breaks: plain `x=1; ready=true` lets the compiler reorder stores;
  observed `garbage=596/200000` on x86 (compiler, not HW, culprit here).
- Fix: `ready.store(release)` + `ready.load(acquire)`; fixed `garbage=0`.
- Run: `g++ -O2 -std=c++17 demo.cpp -o d [-DBROKEN] && ./d`
- Expected: `[x86] FIXED garbage=0/200000 (forbidden per litmus/C++ model; x86 0/200k run only shows the compiler fix)`

## 2. invqueue-loadload
- Concept: two loads must stay ordered (flag, then payload).
- What breaks: relaxed flag load admits stale payload (`r1=1 /\ r2=0` in
  `litmus.txt`); x86 run shows `stale=0` (TSO already orders load-load).
- Fix: acquire flag load (cf. Linux `rmb()`); fixed `stale=0`.
- Run: same pattern as above.
- Expected: `[x86] FIXED stale=0/200000 (forbidden per litmus/C++ model; x86 0/200k run only shows the compiler fix)`

## 3. compiler-barrier-merged (hoist + volatile myth)
- Concept: `-O2` hoists `while(!flag)` load out of the loop; `volatile`
  reloads but is still a data race with no ordering.
- What breaks: broken hits spin CAP instantly (`HOISTED-at-O2`); asm shows one
  `cmpb flag(%rip)` then no reload (`andl $300000000` counter fold).
- Fix: `atomic` acquire load + `barrier()`; asm shows `movzbl flag(%rip)`
  inside `.L17` every iteration; fixed observes in ~50ms.
- Run: same pattern; asm via `g++ -O2 -std=c++17 [-DBROKEN] -S demo.cpp`.
- Expected: `FIXED spins=~3.6e8 ms=50 observed via acquire+barrier`
