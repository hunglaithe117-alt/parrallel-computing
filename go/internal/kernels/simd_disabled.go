//go:build !goexperiment.simd

// Fallback for the experimental archsimd variants. The default build compiles
// this file instead of simd_enabled.go, so `go build ./...` never needs
// GOEXPERIMENT=simd. AVX2FMAEnabled is false and callers skip these functions;
// they panic if reached directly.
package kernels

// AVX2FMAEnabled is false because the archsimd variants were not compiled in.
const AVX2FMAEnabled = false

func VectorAddAVX2FMA(a, b, c []float32) {
	panic("simdlab: avx2_fma requires GOEXPERIMENT=simd")
}

func DotProductAVX2FMA(a, b []float32) float64 {
	panic("simdlab: avx2_fma requires GOEXPERIMENT=simd")
}

func SumReduceAVX2FMA(a []float32) float64 {
	panic("simdlab: avx2_fma requires GOEXPERIMENT=simd")
}
