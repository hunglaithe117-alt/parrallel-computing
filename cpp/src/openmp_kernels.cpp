// OpenMP variants, built with -fopenmp -march=native.
//
// Two very different uses of threads:
//   * mandelbrot_openmp splits independent image rows across threads -- the
//     textbook data-parallel win, because every pixel is independent.
//   * false_sharing_* hammers shared counters to show that threads writing
//     *different* variables can still fight over one cache line.

#include "kernels.h"

#include <omp.h>

#include <cstddef>

// ---------------------------------------------------------------------------
// mandelbrot: parallelize over rows. Rows share nothing but the output buffer,
// and each row owns a disjoint slice of it, so there is no synchronization in
// the loop body and results are identical to the scalar render.
//
// Dynamic scheduling matters on this hybrid CPU: the P-cores are much faster
// than the four E-cores, so a static split leaves fast cores idle at the
// barrier while the E-cores finish their fixed share.
// ---------------------------------------------------------------------------
void mandelbrot_openmp(int w, int h, int max_iter,
                       unsigned char* __restrict out) {
#pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < h; ++y) {
        mb::render_row(w, h, y, max_iter, out);
    }
}

// ---------------------------------------------------------------------------
// false_sharing
//
// Each thread increments only its own element, so there is no logical data
// race. The counters are volatile so the increment cannot be folded into a
// single add, keeping one real read-modify-write per iteration -- that is the
// traffic that lets false sharing show up.
// ---------------------------------------------------------------------------

// The OpenMP runtime already knows OMP_NUM_THREADS and, for recent libgomp, the
// process affinity mask; expose that so the driver does not guess from
// std::thread::hardware_concurrency().
int default_thread_count() {
    const int max_threads = omp_get_max_threads();
    return max_threads > 0 ? max_threads : 1;
}

long false_sharing_adjacent(long iters, int threads,
                            volatile long* __restrict counters) {
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        volatile long* mine = counters + t;  // neighbours share a cache line
        *mine = 0;
        for (long i = 0; i < iters; ++i) {
            *mine = *mine + 1;
        }
    }

    long total = 0;
    for (int t = 0; t < threads; ++t) {
        total += counters[t];
    }
    return total;
}

long false_sharing_padded(long iters, int threads,
                          PaddedCounter* __restrict counters) {
    // alignas(64) makes sizeof(PaddedCounter) == 64, so each counter occupies
    // its own cache line and threads never invalidate a neighbour's line.
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        PaddedCounter* mine = counters + t;
        mine->v = 0;
        for (long i = 0; i < iters; ++i) {
            mine->v = mine->v + 1;
        }
    }

    long total = 0;
    for (int t = 0; t < threads; ++t) {
        total += counters[t].v;
    }
    return total;
}
