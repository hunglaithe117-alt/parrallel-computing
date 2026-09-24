// Hand-written SIMD variants, built with -mavx2 -mfma.
//
// Buffers are 32-byte aligned by bench::make_aligned (see bench.h). Loads and
// stores use the unaligned intrinsics: they are free when the address is in
// fact aligned, and they keep these kernels safe if they are ever pointed at a
// stack buffer. Every kernel handles the n % 8 tail with a scalar remainder
// loop. The tail's arithmetic matches its vector body: plain float add/mul for
// vector_add and saxpy, double accumulation only for dot_product and
// sum_reduce.

#include "kernels.h"

#include <cstddef>
#include <immintrin.h>

// Horizontal sum of the eight lanes of a __m256.
static inline double hsum_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return static_cast<double>(_mm_cvtss_f32(sum));
}

// ---------------------------------------------------------------------------
// vector_add
// ---------------------------------------------------------------------------
void vector_add_sse2(const float* __restrict a, const float* __restrict b,
                     float* __restrict c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; ++i) {  // scalar tail
        c[i] = a[i] + b[i];
    }
}

void vector_add_avx2_fma(const float* __restrict a, const float* __restrict b,
                         float* __restrict c, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) {  // scalar tail
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// saxpy
// ---------------------------------------------------------------------------
void saxpy_avx2_fma(float alpha, const float* __restrict x,
                    float* __restrict y, std::size_t n) {
    const __m256 valpha = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        const __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(valpha, vx, vy));
    }
    for (; i < n; ++i) {  // scalar tail
        y[i] = alpha * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// dot_product
//
// Four independent FMA accumulators give the out-of-order core four separate
// dependency chains to overlap; a single accumulator would serialize the
// multiply-adds. Summing the lanes only at the end keeps that instruction-level
// parallelism. Accumulation is done in float (256-bit = 8 lanes) -- the FMA
// rounding difference from the double-accumulating scalar reference stays
// inside the contract's 1e-4 relative tolerance.
// ---------------------------------------------------------------------------
double dot_product_avx2_fma(const float* __restrict a,
                            const float* __restrict b, std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t i = 0;
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
    for (; i < n; ++i) {  // scalar tail, accumulated in double
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// sum_reduce -- multi-accumulator structure, same reasoning as dot_product.
// ---------------------------------------------------------------------------
double sum_reduce_avx2_fma(const float* __restrict a, std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t i = 0;
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
    for (; i < n; ++i) {  // scalar tail, accumulated in double
        sum += static_cast<double>(a[i]);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// mandelbrot -- eight neighbouring pixels per iteration of the inner loop.
//
// The escape test is branch divergence made explicit: a lane that has escaped
// stops incrementing its counter (masked add) while the other lanes keep
// going; the whole vector stops only once every lane has escaped. That is why
// this kernel is expected to trail the scalar version less than the pure-FLOP
// kernels -- the useful work is uneven across lanes.
// ---------------------------------------------------------------------------
void mandelbrot_avx2(int w, int h, int max_iter,
                     unsigned char* __restrict out) {
    const __m256 vfour = _mm256_set1_ps(4.0f);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vmax = _mm256_set1_ps(255.0f);
    alignas(32) float counts[8];

    for (int y = 0; y < h; ++y) {
        const __m256 vcy = _mm256_set1_ps(mb::cy(w, h, y));
        unsigned char* row = out + static_cast<std::size_t>(y) * w;
        int x = 0;
        for (; x + 8 <= w; x += 8) {
            const __m256 vcx =
                _mm256_set_ps(mb::cx(w, x + 7), mb::cx(w, x + 6),
                              mb::cx(w, x + 5), mb::cx(w, x + 4),
                              mb::cx(w, x + 3), mb::cx(w, x + 2),
                              mb::cx(w, x + 1), mb::cx(w, x + 0));
            __m256 zx = _mm256_setzero_ps();
            __m256 zy = _mm256_setzero_ps();
            __m256 it = _mm256_setzero_ps();

            for (int k = 0; k < max_iter; ++k) {
                const __m256 zx2 = _mm256_mul_ps(zx, zx);
                const __m256 zy2 = _mm256_mul_ps(zy, zy);
                const __m256 mag = _mm256_add_ps(zx2, zy2);
                // Ordered compare: escaped/NaN lanes are inactive.
                const __m256 active = _mm256_cmp_ps(mag, vfour, _CMP_LE_OS);
                it = _mm256_add_ps(it, _mm256_and_ps(active, vone));
                if (_mm256_movemask_ps(active) == 0) {
                    break;
                }
                zy = _mm256_fmadd_ps(_mm256_add_ps(zx, zx), zy, vcy);
                zx = _mm256_add_ps(_mm256_sub_ps(zx2, zy2), vcx);
            }

            const __m256 clamped = _mm256_min_ps(it, vmax);
            _mm256_store_ps(counts, clamped);
            for (int k = 0; k < 8; ++k) {
                row[x + k] = static_cast<unsigned char>(counts[k]);
            }
        }
        for (; x < w; ++x) {  // scalar tail for w % 8 != 0
            row[x] = mb::render_pixel(w, h, x, y, max_iter);
        }
    }
}
