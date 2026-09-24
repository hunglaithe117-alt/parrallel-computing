//go:build goexperiment.simd

// Experimental AVX2/FMA variants, built only with GOEXPERIMENT=simd.
//
// Go 1.27 ships simd/archsimd behind GOEXPERIMENT=simd; it is explicitly
// unstable and not covered by the Go 1 compatibility promise (GA is deferred to
// 1.28). The default `go build` never sees this file, and simd_disabled.go
// provides the same symbols so the default build links without the experiment.
// Float32x8 lowers to 256-bit AVX2 registers; MulAdd lowers to VFMADD213PS.
package kernels

import "simd/archsimd"

// AVX2FMAEnabled is a compile-time constant reporting whether these variants
// were compiled in, so callers can omit the avx2_fma rows otherwise.
const AVX2FMAEnabled = true

// VectorAddAVX2FMA computes c = a + b eight float32 lanes at a time, with a
// scalar tail for n % 8.
func VectorAddAVX2FMA(a, b, c []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8(a[i:]).Add(archsimd.LoadFloat32x8(b[i:])).Store(c[i:])
	}
	for ; i < n; i++ {
		c[i] = a[i] + b[i]
	}
}

// DotProductAVX2FMA uses four independent 256-bit FMA accumulators to break the
// dependency chain, then reduces the lanes into float64. Accumulating in
// float32 rounds differently from the scalar reference; the difference stays
// inside the contract's 1e-4 relative tolerance.
func DotProductAVX2FMA(a, b []float32) float64 {
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	n := len(a)
	i := 0
	for ; i+32 <= n; i += 32 {
		acc0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), acc0)
		acc1 = archsimd.LoadFloat32x8(a[i+8:]).MulAdd(archsimd.LoadFloat32x8(b[i+8:]), acc1)
		acc2 = archsimd.LoadFloat32x8(a[i+16:]).MulAdd(archsimd.LoadFloat32x8(b[i+16:]), acc2)
		acc3 = archsimd.LoadFloat32x8(a[i+24:]).MulAdd(archsimd.LoadFloat32x8(b[i+24:]), acc3)
	}
	for ; i+8 <= n; i += 8 {
		acc0 = archsimd.LoadFloat32x8(a[i:]).MulAdd(archsimd.LoadFloat32x8(b[i:]), acc0)
	}
	var lanes [8]float32
	acc0.Add(acc1).Add(acc2.Add(acc3)).StoreArray(&lanes)
	var sum float64
	for _, v := range lanes {
		sum += float64(v)
	}
	for ; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

// SumReduceAVX2FMA mirrors DotProductAVX2FMA without the multiply.
func SumReduceAVX2FMA(a []float32) float64 {
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	n := len(a)
	i := 0
	for ; i+32 <= n; i += 32 {
		acc0 = acc0.Add(archsimd.LoadFloat32x8(a[i:]))
		acc1 = acc1.Add(archsimd.LoadFloat32x8(a[i+8:]))
		acc2 = acc2.Add(archsimd.LoadFloat32x8(a[i+16:]))
		acc3 = acc3.Add(archsimd.LoadFloat32x8(a[i+24:]))
	}
	for ; i+8 <= n; i += 8 {
		acc0 = acc0.Add(archsimd.LoadFloat32x8(a[i:]))
	}
	var lanes [8]float32
	acc0.Add(acc1).Add(acc2.Add(acc3)).StoreArray(&lanes)
	var sum float64
	for _, v := range lanes {
		sum += float64(v)
	}
	for ; i < n; i++ {
		sum += float64(a[i])
	}
	return sum
}
