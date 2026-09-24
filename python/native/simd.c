/* x86-64 AVX2/FMA kernels for the Python lane's ctypes bridge.
 *
 * Built by the python/Makefile `lib` target:
 *   gcc -O3 -mavx2 -mfma -shared -fPIC -o native/libsimd.so native/simd.c
 *
 * The whole translation unit is compiled with -mavx2 -mfma, so the scalar
 * functions are explicitly decorated to keep GCC's loop and SLP vectorizers off
 * and contracts off: `ctypes_scalar` must be a genuine scalar reference, not a
 * silently auto-vectorized loop. The AVX2 functions use 256-bit unaligned
 * loads, FMA, and a scalar tail for n % 8. Their reduction accumulators are
 * 256-bit float32 (_mm256_fmadd_ps / _mm256_add_ps); only the horizontal fold
 * (hsum_ps) and the scalar tail accumulate in double. That float32 lane
 * accumulation -- not a double vector accumulator -- is the source of the small
 * within-tolerance (1e-4) delta against the float64 pure-Python reference.
 *
 * `size_t` (ctypes c_size_t) keeps the signatures ABI-clean on x86-64.
 */
#include <immintrin.h>
#include <stddef.h>

#if defined(__GNUC__)
#define BENCH_NOVEC                                            \
    __attribute__((optimize("no-tree-vectorize",               \
                            "no-tree-slp-vectorize",           \
                            "fp-contract=off")))
#else
#define BENCH_NOVEC
#endif

/* Horizontal sum of the eight float lanes of a __m256, in double. */
static inline double hsum_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return (double)_mm_cvtss_f32(sum);
}

/* ------------------------------------------------------------------ */
/* vector_add: c[i] = a[i] + b[i]                                      */
/* ------------------------------------------------------------------ */
BENCH_NOVEC
void vector_add_scalar(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_avx2(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) { /* scalar tail */
        c[i] = a[i] + b[i];
    }
}

/* ------------------------------------------------------------------ */
/* dot_product: sum(a[i] * b[i])                                       */
/*                                                                     */
/* Four independent float32 FMA accumulators give the out-of-order core */
/* four dependency chains to overlap; the eight lanes are folded to a   */
/* double once at the end.                                             */
/* ------------------------------------------------------------------ */
BENCH_NOVEC
double dot_product_scalar(const float* a, const float* b, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += (double)a[i] * (double)b[i];
    }
    return acc;
}

double dot_product_avx2(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                               _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                               _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                               _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
    }

    const __m256 lanes =
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps(lanes);
    for (; i < n; ++i) { /* scalar tail, accumulated in double */
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* sum_reduce: sum(a[i])                                               */
/* ------------------------------------------------------------------ */
BENCH_NOVEC
double sum_reduce_scalar(const float* a, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += (double)a[i];
    }
    return acc;
}

double sum_reduce_avx2(const float* a, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(a + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(a + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(a + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(a + i + 24));
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(a + i));
    }

    const __m256 lanes =
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    double sum = hsum_ps(lanes);
    for (; i < n; ++i) { /* scalar tail, accumulated in double */
        sum += (double)a[i];
    }
    return sum;
}
