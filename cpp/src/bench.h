#ifndef PARALLEL_LAB_CPP_BENCH_H_
#define PARALLEL_LAB_CPP_BENCH_H_

// Header-only benchmark harness for the C++ lane.
//
// It implements the frozen output contract in docs/BENCH_CONTRACT.md:
//   * std::chrono::steady_clock timing,
//   * one untimed warm-up pass before measuring,
//   * the median over BENCH_REPS (default 7) timed runs,
//   * size precedence --n <count>, then env BENCH_N, else a per-kernel default,
//   * exactly one "BENCH,<example>,<variant>,<n>,<ms>,<checksum>" line per
//     variant on stdout, ms rendered with 4 decimals,
//   * a volatile sink so the optimizer cannot delete the measured work,
//   * build context and the optional human summary go to stderr only.
//
// Timing excludes allocation, data fill, and checksum computation: callers
// allocate/fill before calling measure() and sum the output afterwards.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace bench {

// Stores through this sink are observable outside the translation unit, so the
// compiler must keep both the kernel result and its checksum accumulation.
inline volatile double g_sink = 0.0;
inline volatile long long g_int_sink = 0;

inline void consume(double value) { g_sink = g_sink + value; }
inline void consume_int(long long value) { g_int_sink = g_int_sink + value; }

struct Config {
    std::string example;
    std::size_t n = 0;
    int reps = 7;
    bool verbose = false;
};

// Human-readable measurement record and dispersion context.
struct Result {
    double ms = 0.0;
    double checksum = 0.0;
    double min_ms = 0.0;
};

inline const char* simd_name() {
#if defined(__AVX512F__)
    return "avx512";
#elif defined(__AVX2__) && defined(__FMA__)
    return "avx2+fma";
#elif defined(__AVX2__)
    return "avx2";
#elif defined(__SSE2__)
    return "sse2";
#else
    return "scalar";
#endif
}

inline int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "bench: ignoring invalid %s='%s'\n", name, raw);
        return fallback;
    }
    return static_cast<int>(value);
}

// Contract v2: n <= 0 is rejected with a clear error, as is a non-numeric or
// negative --reps. strtoll (not strtoull) is used so a leading '-' is caught
// rather than wrapping around to a huge unsigned value.
inline std::size_t parse_positive_size(const char* raw, const char* what) {
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE || value <= 0) {
        std::fprintf(stderr,
                     "bench: %s must be a positive integer, got '%s'\n", what,
                     raw);
        std::exit(2);
    }
    return static_cast<std::size_t>(value);
}

// Parse CLI, resolve the element count from --n / BENCH_N / default_n, and
// announce the build context once to stderr (contract measurement rule 5).
inline Config init(int argc, char** argv, std::size_t default_n,
                   const char* example) {
    Config cfg;
    cfg.example = example;
    cfg.n = default_n;
    cfg.reps = env_int("BENCH_REPS", 7);

    bool size_from_cli = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            cfg.n = parse_positive_size(argv[++i], "--n");
            size_from_cli = true;
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            cfg.reps =
                static_cast<int>(parse_positive_size(argv[++i], "--reps"));
        } else if (std::strcmp(argv[i], "--verbose") == 0 ||
                   std::strcmp(argv[i], "-v") == 0) {
            cfg.verbose = true;
        } else {
            std::fprintf(stderr,
                         "bench: usage: %s [--n <count>] [--reps <count>] "
                         "[--verbose]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    // CLI wins over the environment; env wins over the compiled default.
    if (!size_from_cli) {
        const char* raw = std::getenv("BENCH_N");
        if (raw != nullptr && *raw != '\0') {
            cfg.n = parse_positive_size(raw, "BENCH_N");
        }
    }

    std::fprintf(stderr,
                 "bench[%s]: n=%zu reps=%d compiler=%s simd=%s threads=%u\n",
                 cfg.example.c_str(), cfg.n, cfg.reps, __VERSION__, simd_name(),
                 std::thread::hardware_concurrency());
    return cfg;
}

// Untimed reset hook for in-place kernels. The default is a no-op, so callers
// that do not need it can ignore the parameter.
struct NoReset {
    void operator()() const noexcept {}
};

struct Stats {
    double median_ms = 0.0;
    double min_ms = 0.0;
};

// Time fn() reps times and report the median (contract v2: for an even reps the
// median is the average of the two middle samples) plus the minimum for
// dispersion context. fn must not allocate or perform I/O.
//
// reset() runs untimed after the warm-up and before every timed rep. That makes
// in-place kernels (saxpy) REPS-invariant: the destination is restored from a
// pristine copy each time, so the checksum describes exactly one application
// and the reset never appears in the timed region.
template <typename Fn, typename Reset = NoReset>
inline Stats measure_stats(Fn&& fn, int reps, Reset&& reset = NoReset{}) {
    consume(fn());  // one untimed warm-up pass (contract measurement rule 1)
    reset();        // in-place kernels: undo the warm-up, untimed

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        reset();  // untimed restore, outside the measured region
        const auto t0 = std::chrono::steady_clock::now();
        const double value = fn();
        const auto t1 = std::chrono::steady_clock::now();
        consume(value);
        samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t m = samples.size();
    const double median =
        (m % 2 == 1) ? samples[m / 2]
                     : 0.5 * (samples[m / 2 - 1] + samples[m / 2]);
    return {median, samples.front()};
}

// Convenience wrapper when only the median is needed.
template <typename Fn, typename Reset = NoReset>
inline double measure(Fn&& fn, int reps, Reset&& reset = NoReset{}) {
    return measure_stats(std::forward<Fn>(fn), reps,
                         std::forward<Reset>(reset))
        .median_ms;
}

// The single stdout line the aggregation harness parses.
inline void emit_bench(const char* example, const char* variant, std::size_t n,
                       double ms, double checksum) {
    std::printf("BENCH,%s,%s,%zu,%.4f,%.10g\n", example, variant, n, ms,
                checksum);
    std::fflush(stdout);
}

// Per-variant dispersion context to stderr (contract measurement rule 7).
// The optional pretty summary is appended only under --verbose.
inline void report(const Config& cfg, const char* variant, const Result& r,
                   const char* note = nullptr) {
    std::fprintf(stderr,
                 "bench[%s/%s]: median=%.4f ms min=%.4f ms checksum=%.10g\n",
                 cfg.example.c_str(), variant, r.ms, r.min_ms, r.checksum);
    if (!cfg.verbose) {
        return;
    }
    std::fprintf(stderr, "    %-16s %9.4f ms  # %s\n", variant, r.ms,
                 note != nullptr ? note : "");
}

// Contract v2 numerical tolerance: true relative comparison, denominator
// max(|reference|, 1e-12), never `==`.
inline bool within_tolerance(double value, double reference,
                             double rel_tol = 1e-4) {
    const double scale = std::max(std::fabs(reference), 1e-12);
    return std::fabs(value - reference) / scale <= rel_tol;
}

inline bool check_variant(const char* variant, double value, double reference,
                          double rel_tol = 1e-4) {
    if (within_tolerance(value, reference, rel_tol)) {
        return true;
    }
    std::fprintf(stderr,
                 "bench: CORRECTNESS FAILURE %s: %.10g vs reference %.10g\n",
                 variant, value, reference);
    return false;
}

// 32-byte-aligned heap buffer. alignas(32) cannot apply to a runtime-length
// array, so alignment is requested from the allocator instead. The SIMD kernels
// use _mm256_loadu_ps/_mm256_storeu_ps, which do not require alignment, so this
// is a convenience (cache-line-aligned buffers, and safe if aligned intrinsics
// are added later), not a correctness requirement.
inline constexpr std::size_t kVectorAlignment = 32;

template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { std::free(p); }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T>
inline AlignedPtr<T> make_aligned(std::size_t count,
                                  std::size_t alignment = kVectorAlignment) {
    const std::size_t bytes = count * sizeof(T);
    const std::size_t rounded =
        (bytes + alignment - 1) / alignment * alignment;
    void* raw = std::aligned_alloc(alignment, rounded);
    if (raw == nullptr) {
        std::fprintf(stderr, "bench: %.0f-byte allocation failed\n",
                     static_cast<double>(bytes));
        std::exit(1);
    }
    T* typed = static_cast<T*>(raw);
    if (reinterpret_cast<std::uintptr_t>(typed) % alignment != 0) {
        std::fprintf(stderr, "bench: allocator returned misaligned memory\n");
        std::exit(1);
    }
    return AlignedPtr<T>(typed);
}

// Deterministic positive fill. Positive, same-magnitude data keeps reduction
// checksums large and well-conditioned so the 1e-4 relative tolerance is
// meaningful (a sign-cancelling sum would be dominated by rounding noise).
inline void fill_positive(float* __restrict out, std::size_t n,
                          std::uint32_t seed) {
    std::uint32_t state = seed != 0 ? seed : 0x9e3779b9u;
    for (std::size_t i = 0; i < n; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out[i] = 0.5f + static_cast<float>(state & 0xffffu) / 65535.0f;
    }
}

// Untimed checksum over an output buffer. Compiled once per main translation
// unit, so every variant pays the identical (and unmeasured) checksum cost.
inline double sum_double(const float* __restrict data, std::size_t n) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += static_cast<double>(data[i]);
    }
    return total;
}

inline double sum_bytes(const unsigned char* __restrict data, std::size_t n) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total += static_cast<double>(data[i]);
    }
    return total;
}

}  // namespace bench

#endif  // PARALLEL_LAB_CPP_BENCH_H_
