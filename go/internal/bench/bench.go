// Package bench implements the frozen benchmark output contract
// (docs/BENCH_CONTRACT.md v2.1) for the Go lane:
//
//   - time.Now / time.Since timing of the kernel only,
//   - one untimed warm-up pass before measuring,
//   - the median over BENCH_REPS (default 7) timed runs, averaging the two
//     middle samples for an even count,
//   - n resolved from --n, then BENCH_N, then the per-example default; n <= 0
//     is rejected,
//   - exactly one "BENCH,<example>,<variant>,<n>,<ms>,<checksum>" line per
//     variant on stdout, ms rendered with 4 decimals,
//   - build and thread context always on stderr, plus min and median for every
//     variant (never gated behind a flag),
//   - a package-level Sink so the compiler cannot delete the measured work.
//
// Allocation and the checksum pass happen outside the timed region. Callers
// allocate and fill their buffers before calling Measure.
package bench

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Sink receives every observed result. It is exported and therefore a side
// effect the compiler must preserve: outputs that feed a kernel checksum cannot
// be removed as dead code.
var Sink float64

// Config is the resolved run configuration for one example.
type Config struct {
	Example string
	N       int
	Reps    int
	Verbose bool
}

// Stats is the dispersion context for one variant.
type Stats struct {
	MedianMS float64
	MinMS    float64
}

// Threads reports the worker count the goroutines variants will use, which is
// the current GOMAXPROCS. Multi-thread contract runs set it explicitly
// (GOMAXPROCS=12) and pin with taskset; single-thread runs leave it at the
// default and are understood to collapse every goroutines row to one core.
func Threads() int { return runtime.GOMAXPROCS(0) }

// Init parses the command line and the BENCH_REPS / BENCH_N environment
// variables and announces the build/thread context once to stderr.
func Init(example string, defaultN int) Config {
	cfg := Config{Example: example, N: defaultN, Reps: 7}

	if raw := os.Getenv("BENCH_REPS"); raw != "" {
		if r, err := strconv.Atoi(raw); err == nil && r > 0 {
			cfg.Reps = r
		} else {
			fmt.Fprintf(os.Stderr, "bench: ignoring invalid BENCH_REPS=%q\n", raw)
		}
	}

	sizeFromCLI := false
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--n":
			if i+1 >= len(args) {
				usage()
			}
			cfg.N = positiveInt(args[i+1], "--n")
			sizeFromCLI = true
			i++
		case "--reps":
			if i+1 >= len(args) {
				usage()
			}
			cfg.Reps = positiveInt(args[i+1], "--reps")
			i++
		case "--verbose", "-v":
			cfg.Verbose = true
		default:
			usage()
		}
	}
	if !sizeFromCLI {
		if raw := os.Getenv("BENCH_N"); raw != "" {
			cfg.N = positiveInt(raw, "BENCH_N")
		}
	}

	printContext(cfg)
	return cfg
}

func usage() {
	fmt.Fprintf(os.Stderr,
		"bench: usage: %s [--n <count>] [--reps <count>] [--verbose]\n", os.Args[0])
	os.Exit(2)
}

// positiveInt parses a strictly positive count, rejecting non-numeric and
// n <= 0 values with a clear error (contract requirement).
func positiveInt(raw, what string) int {
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		fmt.Fprintf(os.Stderr, "bench: %s must be a positive integer, got %q\n", what, raw)
		os.Exit(2)
	}
	return value
}

func buildSetting(key string) string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	for _, setting := range info.Settings {
		if setting.Key == key {
			return setting.Value
		}
	}
	return ""
}

// printContext writes the compiler, GOAMD64/SIMD, and thread context to stderr.
// It is called unconditionally: the aggregation harness needs it, not only a
// verbose human.
func printContext(cfg Config) {
	goamd64 := buildSetting("GOAMD64")
	if goamd64 == "" {
		goamd64 = "default"
	}
	simd := "none (gc compiler does not auto-vectorize)"
	if strings.Contains(buildSetting("GOEXPERIMENT"), "simd") {
		simd = "archsimd Float32x8 (GOEXPERIMENT=simd, experimental)"
	}
	fmt.Fprintf(os.Stderr,
		"bench[%s]: n=%d reps=%d compiler=%s os=%s/%s goamd64=%s simd=%s gomaxprocs=%d ncpu=%d\n",
		cfg.Example, cfg.N, cfg.Reps, runtime.Version(), runtime.GOOS,
		runtime.GOARCH, goamd64, simd, runtime.GOMAXPROCS(0), runtime.NumCPU())
}

// Measure runs fn once untimed as a warm-up, then reps times and reports the
// median (average of the two middle samples for an even reps) plus the minimum.
// fn must neither allocate nor perform I/O; only the kernel belongs inside it.
func Measure(reps int, fn func() float64) Stats {
	consume(fn()) // untimed warm-up pass

	samples := make([]float64, reps)
	for r := 0; r < reps; r++ {
		start := time.Now()
		value := fn()
		elapsed := time.Since(start)
		consume(value)
		samples[r] = float64(elapsed.Nanoseconds()) / 1e6
	}

	sort.Float64s(samples)
	mid := len(samples) / 2
	median := samples[mid]
	if len(samples)%2 == 0 {
		median = 0.5 * (samples[mid-1] + samples[mid])
	}
	return Stats{MedianMS: median, MinMS: samples[0]}
}

func consume(value float64) { Sink += value }

// Emit writes the single contractual stdout line. The checksum uses %g with 10
// significant digits, matching the C++ lane's %.10g.
func Emit(example, variant string, n int, ms, checksum float64) {
	fmt.Printf("BENCH,%s,%s,%d,%.4f,%.10g\n", example, variant, n, ms, checksum)
}

// Report writes the per-variant median, min, and checksum to stderr. min is
// always emitted (contract measurement rule 7); the optional explanatory note
// is only shown under --verbose.
func Report(cfg Config, variant string, stats Stats, checksum float64, note string) {
	fmt.Fprintf(os.Stderr,
		"bench[%s/%s]: median=%.4f ms min=%.4f ms checksum=%.10g\n",
		cfg.Example, variant, stats.MedianMS, stats.MinMS, checksum)
	if cfg.Verbose {
		fmt.Fprintf(os.Stderr, "    %-16s %9.4f ms  # %s\n",
			variant, stats.MedianMS, note)
	}
}

// WithinTolerance is the contract's true relative comparison with denominator
// max(|reference|, 1e-12); floats are never compared with ==.
func WithinTolerance(value, reference, relTol float64) bool {
	scale := math.Max(math.Abs(reference), 1e-12)
	return math.Abs(value-reference)/scale <= relTol
}

// CheckVariant reports a correctness failure to stderr when value is outside
// the default 1e-4 tolerance of reference.
func CheckVariant(variant string, value, reference float64) bool {
	if WithinTolerance(value, reference, 1e-4) {
		return true
	}
	fmt.Fprintf(os.Stderr,
		"bench: CORRECTNESS FAILURE %s: %.10g vs reference %.10g\n",
		variant, value, reference)
	return false
}

// FillPositive fills dst with deterministic positive values in [0.5, 1.5).
// Positive, same-magnitude data keeps reduction checksums well-conditioned so
// the 1e-4 relative tolerance is meaningful.
func FillPositive(dst []float32, seed uint32) {
	state := seed
	if state == 0 {
		state = 0x9e3779b9
	}
	for i := range dst {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		dst[i] = 0.5 + float32(state&0xffff)/65535.0
	}
}

// SumFloat32 is the untimed checksum pass over an element-wise output buffer.
func SumFloat32(data []float32) float64 {
	var total float64
	for _, v := range data {
		total += float64(v)
	}
	return total
}
