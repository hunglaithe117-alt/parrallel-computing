// false_sharing_adjacent: packed []int64 counters, one per worker.
//
// Variant goroutines. Neighbouring counters share a cache line, so the workers'
// increments invalidate each other's line and the kernel is much slower than the
// padded layout for the same result. n is the total increment count across
// workers (iters per worker times worker count).
//
// Run: ./bin/false_sharing_adjacent [--n <iters-per-thread>] [--verbose]
package main

import (
	"fmt"
	"os"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultIters = 2000000
	cfg := bench.Init("false_sharing_adjacent", defaultIters)

	threads := bench.Threads()
	iters := int64(cfg.N)
	total := int64(cfg.N) * int64(threads)
	counters := make([]int64, threads)

	var checksum float64
	stats := bench.Measure(cfg.Reps, func() float64 {
		checksum = float64(kernels.FalseSharingAdjacent(iters, threads, counters))
		return checksum
	})
	bench.Emit(cfg.Example, "goroutines", int(total), stats.MedianMS, checksum)
	bench.Report(cfg, "goroutines", stats, checksum, "counters share cache lines")

	if checksum != float64(total) {
		fmt.Fprintf(os.Stderr,
			"bench: CORRECTNESS FAILURE goroutines: %.10g vs expected %.10g\n",
			checksum, float64(total))
		os.Exit(1)
	}
}
