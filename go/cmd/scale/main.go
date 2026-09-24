// scale: GOMAXPROCS scaling for the goroutines reduction variants.
//
// For each worker count 1..min(12, NumCPU) it runs sum_reduce and dot_product in
// a goroutine worker pool and prints the median time and speedup relative to the
// single-thread run. This is the Go-lane view of Amdahl's law: both reductions
// are memory-bandwidth-bound, so the curve flattens well below 12x.
//
// Its output is its own table (not the BENCH contract line), and it is built
// into bin-tools/ rather than bin/ so a `bin/*` glob cannot ingest it. Pin the
// run to the P-cores before measuring, e.g.:
//
//	taskset -c 0-11 ./bin-tools/scale
//
// Run: ./bin-tools/scale [--n <count>] [--reps <count>]
package main

import (
	"fmt"
	"runtime"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultN = 1 << 24 // 16 Mi elements
	cfg := bench.Init("scale", defaultN)

	a := make([]float32, cfg.N)
	b := make([]float32, cfg.N)
	bench.FillPositive(a, 0xdeadbeef)
	bench.FillPositive(b, 0x13579bdf)

	maxThreads := runtime.NumCPU()
	if maxThreads > 12 {
		maxThreads = 12
	}
	original := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(original)

	type row struct {
		threads int
		sumMS   float64
		dotMS   float64
	}
	rows := make([]row, 0, maxThreads)

	for p := 1; p <= maxThreads; p++ {
		runtime.GOMAXPROCS(p)
		partials := make([]float64, p)

		sumStats := bench.Measure(cfg.Reps, func() float64 {
			return kernels.SumReduceGoroutines(a, partials)
		})
		dotStats := bench.Measure(cfg.Reps, func() float64 {
			return kernels.DotProductGoroutines(a, b, partials)
		})
		rows = append(rows, row{threads: p, sumMS: sumStats.MedianMS, dotMS: dotStats.MedianMS})
	}

	fmt.Printf("== GOMAXPROCS scaling: goroutines reduction (n=%d, reps=%d) ==\n",
		cfg.N, cfg.Reps)
	fmt.Printf("%7s  %14s  %8s  %14s  %8s\n",
		"threads", "sum_reduce ms", "speedup", "dot_product ms", "speedup")
	baseSum := rows[0].sumMS
	baseDot := rows[0].dotMS
	for _, r := range rows {
		fmt.Printf("%7d  %14.4f  %8.2f  %14.4f  %8.2f\n",
			r.threads, r.sumMS, baseSum/r.sumMS, r.dotMS, baseDot/r.dotMS)
	}
}
