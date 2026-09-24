// Package kernels holds the Go implementations of every lab kernel.
//
// The scalar functions are plain sequential loops. The gc compiler does not
// auto-vectorize them (https://go.dev/wiki/CompilerOptimizations), so unlike the
// C++ lane there is no separate autovec variant to compare against: scalar is
// the language reference and the honest non-SIMD baseline.
//
// The goroutines functions are worker pools over disjoint output ranges. Each
// worker owns its slice; nothing is appended to a shared slice and the only
// cross-worker state is a preallocated scratch buffer of per-worker partials.
package kernels

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// ---------------------------------------------------------------------------
// vector_add: c[i] = a[i] + b[i]
// ---------------------------------------------------------------------------

// VectorAddScalar is a plain sequential loop; the gc compiler does not
// auto-vectorize it.
func VectorAddScalar(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] + b[i]
	}
}

// VectorAddGoroutines splits [0, n) into contiguous chunks and has one
// goroutine write each chunk, so no two workers touch the same output element.
func VectorAddGoroutines(a, b, c []float32, workers int) {
	n := len(a)
	if workers > n {
		workers = n
	}
	if workers < 1 {
		workers = 1
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				c[i] = a[i] + b[i]
			}
		}(lo, hi)
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// dot_product: sum(a[i] * b[i])
// ---------------------------------------------------------------------------

// DotProductScalar accumulates left to right in float64. Every variant uses a
// float64 accumulator so the comparison stays within the 1e-4 tolerance.
func DotProductScalar(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

// DotProductGoroutines computes one partial sum per worker into the
// preallocated partials scratch (len(partials) is the worker count) and sums
// the partials after the wait. Per-worker ranges are disjoint, so each slot is
// written by exactly one goroutine.
func DotProductGoroutines(a, b []float32, partials []float64) float64 {
	n := len(a)
	workers := len(partials)
	if workers > n {
		workers = n
	}
	if workers < 1 {
		return 0
	}
	for w := 0; w < workers; w++ {
		partials[w] = 0
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			var sum float64
			for i := lo; i < hi; i++ {
				sum += float64(a[i]) * float64(b[i])
			}
			partials[w] = sum
		}(w, lo, hi)
	}
	wg.Wait()
	var total float64
	for w := 0; w < workers; w++ {
		total += partials[w]
	}
	return total
}

// ---------------------------------------------------------------------------
// sum_reduce: sum(a[i])
// ---------------------------------------------------------------------------

// SumReduceScalar accumulates left to right in float64.
func SumReduceScalar(a []float32) float64 {
	var sum float64
	for _, v := range a {
		sum += float64(v)
	}
	return sum
}

// SumReduceGoroutines is the reduction analogue of DotProductGoroutines.
func SumReduceGoroutines(a []float32, partials []float64) float64 {
	n := len(a)
	workers := len(partials)
	if workers > n {
		workers = n
	}
	if workers < 1 {
		return 0
	}
	for w := 0; w < workers; w++ {
		partials[w] = 0
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			var sum float64
			for i := lo; i < hi; i++ {
				sum += float64(a[i])
			}
			partials[w] = sum
		}(w, lo, hi)
	}
	wg.Wait()
	var total float64
	for w := 0; w < workers; w++ {
		total += partials[w]
	}
	return total
}

// ---------------------------------------------------------------------------
// false_sharing
// ---------------------------------------------------------------------------
//
// Both layouts increment one private counter per worker, so they compute the
// same total. They differ only in cache-line placement. Plain int64 loads and
// stores have no volatile qualifier in Go, so the increment is atomic.AddInt64:
// that forces a real read-modify-write to memory every iteration, which is what
// makes cache-line ping-pong visible (a register-resident counter would hide
// the effect).
//
// PaddedCounter is exactly one 64-byte cache line. The compile-time assertion
// below pins the size; NewPaddedCounters additionally aligns the slice base to a
// 64-byte boundary, so the stride-64 elements are each line-aligned regardless
// of what the allocator returns.
type PaddedCounter struct {
	Value int64
	_     [56]byte
}

const paddedCounterSize = int(unsafe.Sizeof(PaddedCounter{}))

// Compile-time assertion: PaddedCounter must stay exactly one cache line.
var _ [64 - paddedCounterSize]byte
var _ [paddedCounterSize - 64]byte

// NewPaddedCounters returns threads counters whose first element starts on a
// 64-byte boundary. Go does not 64-byte-align ordinary allocations, so a plain
// make([]PaddedCounter, threads) gives one line per counter only when the
// allocator happens to return an aligned block. Allocating one extra line of
// bytes and slicing at the first aligned offset makes the isolation
// layout-independent. The offset slices inside raw, so raw stays reachable
// through the returned slice.
func NewPaddedCounters(threads int) []PaddedCounter {
	if threads < 1 {
		return nil
	}
	line := uintptr(paddedCounterSize)
	raw := make([]byte, threads*paddedCounterSize+int(line))
	base := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((line - base%line) % line)
	counters := unsafe.Slice((*PaddedCounter)(unsafe.Pointer(&raw[offset])), threads)
	if uintptr(unsafe.Pointer(&counters[0]))%line != 0 {
		panic("simdlab: padded counters are not cache-line aligned")
	}
	return counters
}

// FalseSharingAdjacent uses a packed []int64: neighbouring counters share a
// cache line, so the workers' atomic increments invalidate each other's line.
func FalseSharingAdjacent(iters int64, threads int, counters []int64) int64 {
	for t := 0; t < threads; t++ {
		atomic.StoreInt64(&counters[t], 0)
	}
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func(t int) {
			defer wg.Done()
			counter := &counters[t]
			for i := int64(0); i < iters; i++ {
				atomic.AddInt64(counter, 1)
			}
		}(t)
	}
	wg.Wait()
	var total int64
	for t := 0; t < threads; t++ {
		total += atomic.LoadInt64(&counters[t])
	}
	return total
}

// FalseSharingPadded gives each counter its own cache line so workers never
// invalidate a neighbour's line.
func FalseSharingPadded(iters int64, threads int, counters []PaddedCounter) int64 {
	for t := 0; t < threads; t++ {
		atomic.StoreInt64(&counters[t].Value, 0)
	}
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func(t int) {
			defer wg.Done()
			counter := &counters[t].Value
			for i := int64(0); i < iters; i++ {
				atomic.AddInt64(counter, 1)
			}
		}(t)
	}
	wg.Wait()
	var total int64
	for t := 0; t < threads; t++ {
		total += atomic.LoadInt64(&counters[t].Value)
	}
	return total
}
