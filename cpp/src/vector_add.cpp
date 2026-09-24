// vector_add: c[i] = a[i] + b[i]
//
// Variants: scalar (no vectorization), autovec (plain loop, compiler decides),
// sse2 (128-bit intrinsics), avx2_fma (256-bit intrinsics).
//
// Run: ./bin/vector_add [--n <count>] [--verbose]

#include <cstddef>

#include "bench.h"
#include "kernels.h"

int main(int argc, char** argv) {
    constexpr std::size_t kDefaultN = 1u << 22;  // 4 Mi elements
    const bench::Config cfg = bench::init(argc, argv, kDefaultN, "vector_add");

    auto a = bench::make_aligned<float>(cfg.n);
    auto b = bench::make_aligned<float>(cfg.n);
    auto c = bench::make_aligned<float>(cfg.n);
    bench::fill_positive(a.get(), cfg.n, 0x1234567u);
    bench::fill_positive(b.get(), cfg.n, 0x89abcdefu);

    bool ok = true;
    double reference = 0.0;
    bool have_reference = false;

    // Element-wise kernel: the timed closure is the write loop only. The
    // checksum is summed afterwards, off the clock (contract v2).
    const auto run = [&](const char* variant, auto&& kernel, const char* note) {
        const bench::Stats stats =
            bench::measure_stats([&] { kernel(); return 0.0; }, cfg.reps);
        const double checksum = bench::sum_double(c.get(), cfg.n);
        bench::emit_bench("vector_add", variant, cfg.n, stats.median_ms,
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

    run("scalar", [&] { vector_add_scalar(a.get(), b.get(), c.get(), cfg.n); },
        "reference");
    run("autovec",
        [&] { vector_add_autovec(a.get(), b.get(), c.get(), cfg.n); },
        "compiler chooses");
    run("sse2", [&] { vector_add_sse2(a.get(), b.get(), c.get(), cfg.n); },
        "128-bit");
    run("avx2_fma",
        [&] { vector_add_avx2_fma(a.get(), b.get(), c.get(), cfg.n); },
        "256-bit");

    return ok ? 0 : 1;
}
