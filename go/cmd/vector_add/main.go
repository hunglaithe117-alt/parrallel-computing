// vector_add: c[i] = a[i] + b[i].
//
// Variants: scalar (Go's non-vectorizing reference loop) and goroutines (worker
// pool over disjoint output ranges), plus the experimental avx2_fma variant when
// built with GOEXPERIMENT=simd.
//
// The timed region is the write loop only. The checksum is summed in a separate
// untimed pass afterwards (contract v2).
//
// Run: ./bin/vector_add [--n <count>] [--verbose]
package main

import (
	"os"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultN = 1 << 22 // 4 Mi elements
	cfg := bench.Init("vector_add", defaultN)

	a := make([]float32, cfg.N)
	b := make([]float32, cfg.N)
	c := make([]float32, cfg.N)
	bench.FillPositive(a, 0x1234567)
	bench.FillPositive(b, 0x89abcdef)

	workers := bench.Threads()
	ok := true
	var reference float64
	haveReference := false

	run := func(variant, note string, kernel func()) {
		stats := bench.Measure(cfg.Reps, func() float64 {
			kernel()
			return 0
		})
		checksum := bench.SumFloat32(c) // untimed checksum pass
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

	run("scalar", "reference; gc does not auto-vectorize",
		func() { kernels.VectorAddScalar(a, b, c) })
	run("goroutines", "worker pool over disjoint ranges",
		func() { kernels.VectorAddGoroutines(a, b, c, workers) })
	if kernels.AVX2FMAEnabled {
		run("avx2_fma", "experimental archsimd Float32x8",
			func() { kernels.VectorAddAVX2FMA(a, b, c) })
	}

	if !ok {
		os.Exit(1)
	}
}
