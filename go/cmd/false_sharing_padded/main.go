// false_sharing_padded: one 64-byte cache line per worker counter.
//
// Variant goroutines. The PaddedCounter layout gives every worker its own cache
// line, so the same increments run without cross-core line invalidation and the
// kernel is markedly faster than false_sharing_adjacent. n is the total
// increment count across workers (iters per worker times worker count).
//
// Run: ./bin/false_sharing_padded [--n <iters-per-thread>] [--verbose]
package main

import (
	"fmt"
	"os"

	"simdlab/internal/bench"
	"simdlab/internal/kernels"
)

func main() {
	const defaultIters = 2000000
	cfg := bench.Init("false_sharing_padded", defaultIters)

	threads := bench.Threads()
	iters := int64(cfg.N)
	total := int64(cfg.N) * int64(threads)
	counters := kernels.NewPaddedCounters(threads)

	var checksum float64
	stats := bench.Measure(cfg.Reps, func() float64 {
		checksum = float64(kernels.FalseSharingPadded(iters, threads, counters))
		return checksum
	})
	bench.Emit(cfg.Example, "goroutines", int(total), stats.MedianMS, checksum)
	bench.Report(cfg, "goroutines", stats, checksum, "one cache line per counter")

	if checksum != float64(total) {
		fmt.Fprintf(os.Stderr,
			"bench: CORRECTNESS FAILURE goroutines: %.10g vs expected %.10g\n",
			checksum, float64(total))
		os.Exit(1)
	}
}
