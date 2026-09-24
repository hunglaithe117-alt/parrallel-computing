// rcu-INSPIRED (no grace period — GC + COW), not true RCU.
// Readers load atomic.Pointer without locking; writer swaps a fresh copy.
// True RCU waits for a grace period before reclaim; here GC reclaims the old
// copy once no reader references it.
// Epoch alternative: true userspace RCU needs grace periods (crossbeam-epoch/EBR).
package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

type config struct{ version int }

const readers = 8
const reads = 200000

// sink accumulates loaded versions so the compiler cannot drop the reads.
// Re-measured with sink (Gate 3): RWMutex ~45ms (avg ~28ns/read), COW
// ~0.25ms (avg ~0.2ns/read) wall-clock throughput for 1.6M reads on
// 8 threads; ratios host-dependent.
var sink atomic.Int64

func main() {
	fmt.Println("baseline: RWMutex read-heavy, cow: atomic.Pointer[config]")
	base, bAvg := baseline()
	cow, cAvg := cow()
	fmt.Printf("RWMutex: %d reads in %v (avg %.1fns/read)\n", readers*reads, base, bAvg)
	fmt.Printf("COW:     %d reads in %v (avg %.1fns/read)\n", readers*reads, cow, cAvg)
	fmt.Println("sink (proves loads executed):", sink.Load())
}

// fanout runs fn once per reader thread, collecting each thread's partial sum.
func fanout(fn func() int) []int {
	sums := make([]int, readers)
	var wg sync.WaitGroup
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sums[id] = fn()
		}(i)
	}
	wg.Wait()
	return sums
}

func total(sums []int) int {
	t := 0
	for _, s := range sums {
		t += s
	}
	return t
}

func baseline() (time.Duration, float64) {
	var mu sync.RWMutex
	var c config
	start := time.Now()
	wrote := make(chan struct{})
	go func() { // one concurrent writer shows RLock/Lock exclusion cost
		defer close(wrote)
		for k := 0; k < 100; k++ {
			mu.Lock()
			c.version++
			mu.Unlock()
		}
	}()
	sums := fanout(func() int {
		s := 0
		for j := 0; j < reads; j++ {
			mu.RLock()
			s += c.version // sink: consume the load so DCE cannot drop it
			mu.RUnlock()
		}
		return s
	})
	<-wrote
	sink.Add(int64(total(sums)))
	el := time.Since(start)
	return el, float64(el.Nanoseconds()) / float64(readers*reads)
}

func cow() (time.Duration, float64) {
	var p atomic.Pointer[config]
	p.Store(&config{})
	start := time.Now()
	swapped := make(chan struct{})
	go func() { // concurrent copy-on-write swaps, readers never block
		defer close(swapped)
		for k := 0; k < 100; k++ {
			old := p.Load()
			nw := *old
			nw.version++
			p.Store(&nw)
		}
	}()
	sums := fanout(func() int {
		s := 0
		for j := 0; j < reads; j++ {
			s += p.Load().version // sink: consume the load so DCE cannot drop it
		}
		return s
	})
	<-swapped
	sink.Add(int64(total(sums)))
	el := time.Since(start)
	// Throughput comparison only (avg ns/read); no per-read latency histogram,
	// so no p99 is reported.
	return el, float64(el.Nanoseconds()) / float64(readers*reads)
}
