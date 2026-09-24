// mandelbrot: render W x H, checksum = sum of iteration counts.
//
// Variants: scalar, avx2_fma (eight pixels per inner iteration), openmp
// (rows split across threads). Compute-bound and branch-divergent, which makes
// it a deliberate contrast to the memory-bound vector kernels.
//
// The --n override is read as a total pixel count; width stays 1024 and height
// becomes n / 1024 (capped at >= 1). The emitted n is the actual W*H.
//
// Run: ./bin/mandelbrot [--n <count>] [--verbose]

#include <cstddef>
#include <vector>

#include "bench.h"
#include "kernels.h"

int main(int argc, char** argv) {
    constexpr int kWidth = 1024;
    constexpr int kDefaultHeight = 768;
    constexpr int kMaxIter = 256;
    constexpr std::size_t kDefaultN =
        static_cast<std::size_t>(kWidth) * kDefaultHeight;
    const bench::Config cfg = bench::init(argc, argv, kDefaultN, "mandelbrot");

    const int w = kWidth;
    const int h = cfg.n > static_cast<std::size_t>(kWidth)
                      ? static_cast<int>(cfg.n / kWidth)
                      : 1;
    const std::size_t pixels = static_cast<std::size_t>(w) * h;

    std::vector<unsigned char> image(pixels, 0);

    bool ok = true;
    double reference = 0.0;
    bool have_reference = false;

    // Output kernel: timed region is the render; checksum is summed untimed.
    const auto run = [&](const char* variant, auto&& kernel, const char* note) {
        const bench::Stats stats =
            bench::measure_stats([&] { kernel(); return 0.0; }, cfg.reps);
        const double checksum = bench::sum_bytes(image.data(), pixels);
        bench::emit_bench("mandelbrot", variant, pixels, stats.median_ms,
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

    run("scalar",
        [&] { mandelbrot_scalar(w, h, kMaxIter, image.data()); },
        "reference");
    run("avx2_fma",
        [&] { mandelbrot_avx2(w, h, kMaxIter, image.data()); },
        "8 pixels / iteration");
    run("openmp",
        [&] { mandelbrot_openmp(w, h, kMaxIter, image.data()); },
        "rows across threads");

    return ok ? 0 : 1;
}
