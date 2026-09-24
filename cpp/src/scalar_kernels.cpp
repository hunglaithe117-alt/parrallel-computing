// Reference (scalar) variants.
//
// Built with -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off so
// these loops stay left-to-right scalar operations. They are the checksum
// reference every other variant must match within the contract tolerance.

#include "kernels.h"

#include <cstddef>

// ---------------------------------------------------------------------------
// vector_add
// ---------------------------------------------------------------------------
void vector_add_scalar(const float* __restrict a, const float* __restrict b,
                       float* __restrict c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// saxpy
// ---------------------------------------------------------------------------
void saxpy_scalar(float alpha, const float* __restrict x,
                  float* __restrict y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// dot_product / sum_reduce
//
// Each float operand is widened to double and the product is formed in double,
// then accumulated in a double left-to-right chain, so this stays the exact
// reference. A plain `sum += a[i]*b[i]` reduction cannot be reassociated
// without -ffast-math, which is why the autovec unit keeps the same serial
// order. See the comment on dot_product_autovec.
// ---------------------------------------------------------------------------
double dot_product_scalar(const float* __restrict a,
                          const float* __restrict b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double sum_reduce_scalar(const float* __restrict a, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// mandelbrot (single-threaded, scalar per pixel)
// ---------------------------------------------------------------------------
void mandelbrot_scalar(int w, int h, int max_iter,
                       unsigned char* __restrict out) {
    for (int y = 0; y < h; ++y) {
        mb::render_row(w, h, y, max_iter, out);
    }
}
