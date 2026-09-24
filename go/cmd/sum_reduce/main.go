// sum_reduce: sum(a[i]).
//
// Variants: scalar, goroutines, and the experimental avx2_fma variant when built
// with GOEXPERIMENT=simd. As with dot_product, the reduction is the kernel and
// its float64 result is the checksum (contract v2).
//
// Run: ./bin/sum_reduce [--n <count>] [--verbose]
package main

import (
	"os"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultN = 1 << 24 // 16 Mi elements
	cfg := bench.Init("sum_reduce", defaultN)

	a := make([]float32, cfg.N)
	bench.FillPositive(a, 0xdeadbeef)

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
		func() float64 { return kernels.SumReduceScalar(a) })
	run("goroutines", "one partial sum per worker",
		func() float64 { return kernels.SumReduceGoroutines(a, partials) })
	if kernels.AVX2FMAEnabled {
		run("avx2_fma", "experimental archsimd, four accumulators",
			func() float64 { return kernels.SumReduceAVX2FMA(a) })
	}

	if !ok {
		os.Exit(1)
	}
}
