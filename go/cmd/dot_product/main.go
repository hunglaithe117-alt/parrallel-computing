// dot_product: sum(a[i] * b[i]).
//
// Variants: scalar, goroutines, and the experimental avx2_fma variant when built
// with GOEXPERIMENT=simd. The reduction is the kernel: its float64 result is
// produced inside the timed region and is the checksum (contract v2).
//
// Run: ./bin/dot_product [--n <count>] [--verbose]
package main

import (
	"os"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultN = 1 << 22 // 4 Mi elements
	cfg := bench.Init("dot_product", defaultN)

	a := make([]float32, cfg.N)
	b := make([]float32, cfg.N)
	bench.FillPositive(a, 0x13579bdf)
	bench.FillPositive(b, 0x2468ace0)

	// Partial scratch is preallocated so the goroutines kernel allocates nothing
	// inside the timed region.
	partials := make([]float64, bench.Threads())

	ok := true
	var reference float64
	haveReference := false

	run := func(variant, note string, kernel func() float64) {
		var checksum float64
		stats := bench.Measure(cfg.Reps, func() float64 {
			checksum = kernel()
			return checksum
		})
		bench.Emit(cfg.Example, variant, cfg.N, stats.MedianMS, checksum)
		bench.Report(cfg, variant, stats, checksum, note)
		if !haveReference {
			reference = checksum
			haveReference = true
			return
		}
		if !bench.CheckVariant(variant, checksum, reference) {
			ok = false
		}
	}

	run("scalar", "reference; left-to-right float64 accumulation",
		func() float64 { return kernels.DotProductScalar(a, b) })
	run("goroutines", "one partial sum per worker",
		func() float64 { return kernels.DotProductGoroutines(a, b, partials) })
	if kernels.AVX2FMAEnabled {
		run("avx2_fma", "experimental archsimd, four FMA accumulators",
			func() float64 { return kernels.DotProductAVX2FMA(a, b) })
	}

	if !ok {
		os.Exit(1)
	}
}
