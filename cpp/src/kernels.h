#ifndef PARALLEL_LAB_CPP_KERNELS_H_
#define PARALLEL_LAB_CPP_KERNELS_H_

// Declarations for every benchmark variant.
//
// Variants live in separate translation units grouped by the compiler flags
// they need (see Makefile):
//   scalar_kernels.cpp  -O3 -fno-tree-vectorize -fno-tree-slp-vectorize
//   autovec_kernels.cpp -O3 -march=native          (plain loops)
//   simd_kernels.cpp    -O3 -mavx2 -mfma           (hand-written intrinsics)
//   openmp_kernels.cpp  -O3 -march=native -fopenmp
//
// Splitting the units is what makes "scalar" honestly scalar: with the
// vectorizers disabled the reference cannot be auto-vectorized behind our back.

#include <cstddef>

// ---------------------------------------------------------------------------
// vector_add: c[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
void vector_add_scalar(const float* __restrict a, const float* __restrict b,
                       float* __restrict c, std::size_t n);
void vector_add_autovec(const float* __restrict a, const float* __restrict b,
                        float* __restrict c, std::size_t n);
void vector_add_sse2(const float* __restrict a, const float* __restrict b,
                     float* __restrict c, std::size_t n);
void vector_add_avx2_fma(const float* __restrict a, const float* __restrict b,
                         float* __restrict c, std::size_t n);

// ---------------------------------------------------------------------------
// saxpy: y[i] = alpha * x[i] + y[i]
// ---------------------------------------------------------------------------
void saxpy_scalar(float alpha, const float* __restrict x,
                  float* __restrict y, std::size_t n);
void saxpy_autovec(float alpha, const float* __restrict x,
                   float* __restrict y, std::size_t n);
void saxpy_avx2_fma(float alpha, const float* __restrict x,
                    float* __restrict y, std::size_t n);

// ---------------------------------------------------------------------------
// dot_product: sum(a[i] * b[i])
// ---------------------------------------------------------------------------
double dot_product_scalar(const float* __restrict a,
                          const float* __restrict b, std::size_t n);
double dot_product_autovec(const float* __restrict a,
                           const float* __restrict b, std::size_t n);
double dot_product_avx2_fma(const float* __restrict a,
                            const float* __restrict b, std::size_t n);

// ---------------------------------------------------------------------------
// sum_reduce: sum(a[i])
// ---------------------------------------------------------------------------
double sum_reduce_scalar(const float* __restrict a, std::size_t n);
double sum_reduce_autovec(const float* __restrict a, std::size_t n);
double sum_reduce_avx2_fma(const float* __restrict a, std::size_t n);

// ---------------------------------------------------------------------------
// mandelbrot: render w x h, one iteration count per pixel
// ---------------------------------------------------------------------------
void mandelbrot_scalar(int w, int h, int max_iter,
                       unsigned char* __restrict out);
void mandelbrot_avx2(int w, int h, int max_iter, unsigned char* __restrict out);
void mandelbrot_openmp(int w, int h, int max_iter,
                       unsigned char* __restrict out);

// Shared viewport and scalar pixel math, so the scalar and vectorized renders
// agree on which pixel is which.
namespace mb {

inline constexpr float kXMin = -2.0f;
inline constexpr float kXSpan = 3.0f;

inline float aspect(int w, int h) {
    return static_cast<float>(h) / static_cast<float>(w);
}

inline float cx(int w, int x) {
    return kXMin + (static_cast<float>(x) + 0.5f) * kXSpan /
                       static_cast<float>(w);
}

inline float cy(int w, int h, int y) {
    const float span = kXSpan * aspect(w, h);
    return -0.5f * span +
           (static_cast<float>(y) + 0.5f) * span / static_cast<float>(h);
}

// Iteration count for one pixel, clamped to 255 so it fits one byte.
inline unsigned char render_pixel(int w, int h, int x, int y, int max_iter) {
    const float px = cx(w, x);
    const float py = cy(w, h, y);
    float zx = 0.0f;
    float zy = 0.0f;
    int it = 0;
    while (it < max_iter) {
        const float zx2 = zx * zx;
        const float zy2 = zy * zy;
        if (zx2 + zy2 > 4.0f) {
            break;
        }
        zy = 2.0f * zx * zy + py;
        zx = zx2 - zy2 + px;
        ++it;
    }
    return static_cast<unsigned char>(it < 255 ? it : 255);
}

// One whole row. Used by both the scalar and the OpenMP variant; each
// translation unit compiles this body with its own flags.
inline void render_row(int w, int h, int y, int max_iter,
                       unsigned char* __restrict out) {
    unsigned char* row = out + static_cast<std::size_t>(y) * w;
    for (int x = 0; x < w; ++x) {
        row[x] = render_pixel(w, h, x, y, max_iter);
    }
}

}  // namespace mb

// ---------------------------------------------------------------------------
// false_sharing: N increments per thread
// ---------------------------------------------------------------------------
// PaddedCounter is one full cache line per counter, so neighbouring threads
// never invalidate each other's line.
struct alignas(64) PaddedCounter {
    volatile long v;
};

long false_sharing_adjacent(long iters, int threads,
                            volatile long* __restrict counters);
long false_sharing_padded(long iters, int threads,
                          PaddedCounter* __restrict counters);

// Thread-pool size for the OpenMP kernels. Respects OMP_NUM_THREADS and the
// process CPU affinity; callers may override it with the THREADS env var.
int default_thread_count();

#endif  // PARALLEL_LAB_CPP_KERNELS_H_
