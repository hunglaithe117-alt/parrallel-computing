# C++ memory-model demos

Three minimal (<80-line) demos, each one file with `-DBROKEN` selecting the
broken mode. x86 loop output for the HW demos (storebuffer, invqueue) is a
**negative control** (TSO hides HW reordering; litmus files carry the
weak-model reasoning, NOT machine-verified). The compiler demo's broken mode
is a **[positive control x86: hoist observable]** on this host.

| demo | run |
|---|---|
| storebuffer-visibility | `g++ -O2 -std=c++17 demo.cpp -o d && ./d` |
| invqueue-loadload | same |
| compiler-barrier-merged | same; `-S` asm in `asm-note.txt` |

Or `make -C sync-learn/cpp` to build + run all six modes.
