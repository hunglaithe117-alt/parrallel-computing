// sum_reduce: sum(a[i])
//
// Variants: scalar, autovec, avx2_fma (four independent accumulators).
// As with dot_product, the plain reduction resists auto-vectorization.
//
// Run: ./bin/sum_reduce [--n <count>] [--verbose]

#include <cstddef>

#include "bench.h"
#include "kernels.h"

int main(int argc, char** argv) {
    constexpr std::size_t kDefaultN = 1u << 24;  // 16 Mi elements
    const bench::Config cfg = bench::init(argc, argv, kDefaultN, "sum_reduce");

    auto a = bench::make_aligned<float>(cfg.n);
    bench::fill_positive(a.get(), cfg.n, 0xdeadbeefu);

    bool ok = true;
    double reference = 0.0;
    bool have_reference = false;

    // Reduction kernel: reduction inside the timed region, result is checksum.
    const auto run = [&](const char* variant, auto&& kernel, const char* note) {
        double checksum = 0.0;
        const bench::Stats stats = bench::measure_stats(
            [&] {
                checksum = kernel();
                return checksum;
            },
            cfg.reps);
        bench::emit_bench("sum_reduce", variant, cfg.n, stats.median_ms,
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

    run("scalar", [&] { return sum_reduce_scalar(a.get(), cfg.n); },
        "reference");
    run("autovec", [&] { return sum_reduce_autovec(a.get(), cfg.n); },
        "serial reduction");
    run("avx2_fma", [&] { return sum_reduce_avx2_fma(a.get(), cfg.n); },
        "4 accumulators");

    return ok ? 0 : 1;
}
