// saxpy: y[i] = alpha * x[i] + y[i]
//
// Variants: scalar, autovec, avx2_fma (fused multiply-add).
//
// Run: ./bin/saxpy [--n <count>] [--verbose]

#include <cstddef>
#include <cstring>

#include "bench.h"
#include "kernels.h"

int main(int argc, char** argv) {
    constexpr std::size_t kDefaultN = 1u << 22;  // 4 Mi elements
    constexpr float kAlpha = 0.7f;  // non-dyadic: FMA vs mul+add differ in rounding
    const bench::Config cfg = bench::init(argc, argv, kDefaultN, "saxpy");

    auto x = bench::make_aligned<float>(cfg.n);
    auto y = bench::make_aligned<float>(cfg.n);
    auto y0 = bench::make_aligned<float>(cfg.n);
    bench::fill_positive(x.get(), cfg.n, 0x0badf00du);
    bench::fill_positive(y0.get(), cfg.n, 0xfeedfaceu);

    const std::size_t bytes = cfg.n * sizeof(float);
    bool ok = true;
    double reference = 0.0;
    bool have_reference = false;

    // saxpy is in-place, so measure_stats restores y from the pristine y0 copy
    // before every timed rep. The reset runs untimed, and the checksum is a
    // single application, so the result is invariant to BENCH_REPS (contract v2).
    const auto run = [&](const char* variant, auto&& kernel, const char* note) {
        const bench::Stats stats = bench::measure_stats(
            [&] { kernel(); return 0.0; }, cfg.reps,
            [&] { std::memcpy(y.get(), y0.get(), bytes); });
        const double checksum = bench::sum_double(y.get(), cfg.n);
        bench::emit_bench("saxpy", variant, cfg.n, stats.median_ms, checksum);
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

    run("scalar", [&] { saxpy_scalar(kAlpha, x.get(), y.get(), cfg.n); },
        "reference");
    run("autovec", [&] { saxpy_autovec(kAlpha, x.get(), y.get(), cfg.n); },
        "compiler chooses");
    run("avx2_fma", [&] { saxpy_avx2_fma(kAlpha, x.get(), y.get(), cfg.n); },
        "vfmadd");

    return ok ? 0 : 1;
}
