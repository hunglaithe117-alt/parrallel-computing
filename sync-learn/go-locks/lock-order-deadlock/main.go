package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

var a, b sync.Mutex

func main() {
	mode := "deadlock"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	done := make(chan struct{}, 2)
	if mode == "fixed" {
		go lockAB("g1", done) // global order A->B in both goroutines
		go lockAB("g2", done)
	} else {
		go func() { // g1: A->B
			a.Lock()
			defer a.Unlock()
			time.Sleep(100 * time.Millisecond)
			b.Lock()
			defer b.Unlock()
			fmt.Println("g1 done")
			done <- struct{}{}
		}()
		go func() { // g2: B->A (opposite order)
			b.Lock()
			defer b.Unlock()
			time.Sleep(100 * time.Millisecond)
			a.Lock()
			defer a.Unlock()
			fmt.Println("g2 done")
			done <- struct{}{}
		}()
	}
	for i := 0; i < 2; i++ {
		select {
		case <-done:
		case <-time.After(time.Second):
			fmt.Println("DEADLOCK: g1 holds A wants B, g2 holds B wants A")
			return
		}
	}
	fmt.Println("FIXED: both goroutines completed under global order A->B")
}

func lockAB(name string, done chan struct{}) {
	a.Lock()
	defer a.Unlock()
	time.Sleep(100 * time.Millisecond)
	b.Lock()
	defer b.Unlock()
	fmt.Println(name, "done")
	done <- struct{}{}
}
