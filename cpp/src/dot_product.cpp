// dot_product: sum(a[i] * b[i])
//
// Variants: scalar, autovec, avx2_fma (four independent FMA accumulators).
//
// The interesting result here is that `autovec` gives no speedup: GCC
// vectorizes the multiplies but keeps the reduction in order with scalar adds,
// so the dependency chain remains. The hand-written variant breaks the chain
// with four independent accumulators.
//
// Run: ./bin/dot_product [--n <count>] [--verbose]

#include <cstddef>

#include "bench.h"
#include "kernels.h"

int main(int argc, char** argv) {
    constexpr std::size_t kDefaultN = 1u << 22;  // 4 Mi elements
    const bench::Config cfg = bench::init(argc, argv, kDefaultN, "dot_product");

    auto a = bench::make_aligned<float>(cfg.n);
    auto b = bench::make_aligned<float>(cfg.n);
    bench::fill_positive(a.get(), cfg.n, 0x13579bdfu);
    bench::fill_positive(b.get(), cfg.n, 0x2468ace0u);

    bool ok = true;
    double reference = 0.0;
    bool have_reference = false;

    // Reduction kernel: the reduction is the timed region and its scalar result
    // is the checksum (contract v2).
    const auto run = [&](const char* variant, auto&& kernel, const char* note) {
        double checksum = 0.0;
        const bench::Stats stats = bench::measure_stats(
            [&] {
                checksum = kernel();
                return checksum;
            },
            cfg.reps);
        bench::emit_bench("dot_product", variant, cfg.n, stats.median_ms,
                          checksum);
        bench::report(
            cfg, variant,
            bench::Result{stats.median_ms, checksum, stats.min_ms}, note);
        if (!have_reference) {
            reference = checksum;
            have_reference = true;
        } else if (!bench::check_variant(variant, checksum, reference)) {
            ok = false;
        }
    };

    run("scalar", [&] { return dot_product_scalar(a.get(), b.get(), cfg.n); },
        "reference");
    run("autovec",
        [&] { return dot_product_autovec(a.get(), b.get(), cfg.n); },
        "serial reduction");
    run("avx2_fma",
        [&] { return dot_product_avx2_fma(a.get(), b.get(), cfg.n); },
        "4 accumulators");

    return ok ? 0 : 1;
}
