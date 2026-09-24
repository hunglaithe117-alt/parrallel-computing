// Auto-vectorized variants: identical source to the scalar loops, but this
// translation unit is built with -march=native and the vectorizers enabled.
//
// This is the "let the compiler do it" lane. For element-wise work (vector_add,
// saxpy) GCC emits AVX2 automatically. For reductions (dot_product,
// sum_reduce) it vectorizes the loads and multiplies but must preserve the
// summation order under strict IEEE rules, so the adds stay a serial scalar
// chain and the loop gets almost no SIMD speedup. Breaking that chain by hand
// is the lesson of the avx2_fma variants.

#include "kernels.h"

#include <cstddef>

// ---------------------------------------------------------------------------
// vector_add -- auto-vectorizes to 256-bit AVX2 on this host.
// ---------------------------------------------------------------------------
void vector_add_autovec(const float* __restrict a, const float* __restrict b,
                        float* __restrict c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// saxpy -- auto-vectorizes; GCC contracts the multiply-add into FMA.
// ---------------------------------------------------------------------------
void saxpy_autovec(float alpha, const float* __restrict x,
                   float* __restrict y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// dot_product
//
// A naive reduction is a serial dependency chain: each `sum` update depends on
// the previous one. GCC 15 does vectorize the loads and the multiplies, but it
// cannot reassociate the sum without -ffast-math -- the disassembly shows a
// vector multiply feeding a chain of scalar `vaddsd`. The checksum is therefore
// bit-identical to scalar and there is no speedup. Independent accumulators
// have to be introduced by hand: see dot_product_avx2_fma.
// ---------------------------------------------------------------------------
double dot_product_autovec(const float* __restrict a,
                           const float* __restrict b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// sum_reduce -- same serial-reduction limitation as dot_product: the vector
// units load eight floats but the additions remain scalar and ordered.
// ---------------------------------------------------------------------------
double sum_reduce_autovec(const float* __restrict a, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]);
    }
    return sum;
}
