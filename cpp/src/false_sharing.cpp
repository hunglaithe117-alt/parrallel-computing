// false_sharing: N increments per thread, two memory layouts.
//
// Contract v2.1 names these as two examples sharing the canonical `openmp`
// variant:
//   false_sharing_adjacent - volatile long counters[threads] (8/line)
//   false_sharing_padded   - alignas(64) PaddedCounter[threads] (1/line)
//
// The thread count comes from the THREADS env var, falling back to
// omp_get_max_threads() (which honours OMP_NUM_THREADS and affinity). That is
// the same explicit pool size the other language lanes use, so
// n = iters * threads is comparable across languages.
//
// Both layouts compute the same sum; the only difference is cache-line
// placement, so any timing gap is the false-sharing effect.
//
// Run: ./bin/false_sharing [--n <increments-per-thread>] [--verbose]
//      (also built as bin/false_sharing_adjacent and bin/false_sharing_padded,
//       each of which emits only its own example)

#include <cstddef>
#include <cstring>
#include <vector>

#include "bench.h"
#include "kernels.h"

namespace {

const char* program_basename(const char* argv0) {
    const char* base = std::strrchr(argv0, '/');
    return base != nullptr ? base + 1 : argv0;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr std::size_t kDefaultIters = 2000000;
    const bench::Config cfg =
        bench::init(argc, argv, kDefaultIters, "false_sharing");

    // The alias binaries (same translation unit) emit one example each; the
    // umbrella binary emits both. `make run` uses the umbrella only.
    const char* program = program_basename(argv[0]);
    const bool only_adjacent = std::strstr(program, "adjacent") != nullptr;
    const bool only_padded = std::strstr(program, "padded") != nullptr;
    const bool run_all = !only_adjacent && !only_padded;

    // Explicit THREADS wins; otherwise the OpenMP runtime's configured maximum
    // (OMP_NUM_THREADS / affinity), never raw std::thread::hardware_concurrency.
    const int threads = bench::env_int("THREADS", default_thread_count());
    const long iters = static_cast<long>(cfg.n);
    const double expected = static_cast<double>(iters) * threads;

    // n reported to the harness is the total increment count across threads.
    const std::size_t total = static_cast<std::size_t>(iters) * threads;

    std::vector<long> adjacent(static_cast<std::size_t>(threads), 0);
    std::vector<PaddedCounter> padded(static_cast<std::size_t>(threads));

    if (cfg.verbose) {
        std::fprintf(stderr, "  runs: %s%s%s\n",
                     run_all ? "both" : "", only_adjacent ? "adjacent" : "",
                     only_padded ? "padded" : "");
    }
    // Contract v2.1: state the multi-thread pool size explicitly and
    // unconditionally. The `threads=` field on the context line above is host
    // capacity (hardware_concurrency); this is the benchmark's actual pool.
    std::fprintf(stderr,
                 "bench[false_sharing]: pool_threads=%d iters/thread=%ld "
                 "n=%zu expected=%.0f\n",
                 threads, iters, total, expected);

    bool ok = true;
    const auto run = [&](const char* example, auto&& kernel, const char* note) {
        double checksum = 0.0;
        const bench::Stats stats = bench::measure_stats(
            [&] {
                checksum = static_cast<double>(kernel());
                return checksum;
            },
            cfg.reps);
        bench::emit_bench(example, "openmp", total, stats.median_ms, checksum);
        bench::report(
            cfg, example,
            bench::Result{stats.median_ms, checksum, stats.min_ms}, note);
        if (!bench::check_variant(example, checksum, expected)) {
            ok = false;
        }
    };

    if (run_all || only_adjacent) {
        run("false_sharing_adjacent",
            [&] {
                return false_sharing_adjacent(iters, threads, adjacent.data());
            },
            "counters share cache lines");
    }
    if (run_all || only_padded) {
        run("false_sharing_padded",
            [&] { return false_sharing_padded(iters, threads, padded.data()); },
            "one cache line per counter");
    }

    return ok ? 0 : 1;
}
