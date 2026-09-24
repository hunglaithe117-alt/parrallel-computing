# C++ lane of the parallel-computing lab.
#
# Per-variant flags are the whole point of this Makefile: the "scalar" kernels
# are compiled with the vectorizers disabled, the "autovec" kernels with
# -march=native, the hand-written SIMD kernels with explicit -mavx2 -mfma, and
# the OpenMP kernels with -fopenmp. Because flags are per translation unit, the
# variants live in separate .cpp files and are linked into one binary per
# benchmark (kernels.h declares the shared interface).

# Contract v2 stream hygiene: stdout carries only BENCH, lines. Suppress make's
# "Entering/Leaving directory" banners (implied by -C) and recipe echoing.
MAKEFLAGS += --no-print-directory
.SILENT:

CXX ?= g++
CXXSTD := -std=c++17
WARN := -Wall -Wextra -Wpedantic
OPT := -O3
COMMON := $(OPT) $(CXXSTD) $(WARN)

# No -ffast-math anywhere in the default build: the checksums must stay
# comparable with the contract's 1e-4 relative tolerance.
SCALAR_FLAGS := $(COMMON) -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off
AUTOVEC_FLAGS := $(COMMON) -march=native
SIMD_FLAGS := $(COMMON) -mavx2 -mfma
OMP_FLAGS := $(COMMON) -march=native -fopenmp
# Mains only orchestrate: allocate, time, checksum. Built native so the
# stderr build-context line reports the real SIMD level and so the untimed
# checksum loops are identical for every variant.
MAIN_FLAGS := $(COMMON) -march=native
LINK_FLAGS := -fopenmp

BUILD := build
BIN := bin

GROUP_SRCS := src/scalar_kernels.cpp src/autovec_kernels.cpp \
              src/simd_kernels.cpp src/openmp_kernels.cpp
GROUP_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(GROUP_SRCS))

KERNELS := vector_add saxpy dot_product sum_reduce mandelbrot false_sharing
MAIN_SRCS := $(addprefix src/,$(addsuffix .cpp,$(KERNELS)))
MAIN_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(MAIN_SRCS))
BINS := $(addprefix $(BIN)/,$(KERNELS))

# Per-example aliases for false_sharing: same binary, but invoked under these
# names it emits only the matching example (see src/false_sharing.cpp). Keeps
# `make run` on the umbrella binary free of duplicate BENCH rows.
ALIASES := $(BIN)/false_sharing_adjacent $(BIN)/false_sharing_padded

# Keep the main objects around; without this make treats them as intermediate
# and deletes them, forcing a rebuild on every `make run`.
.SECONDARY: $(MAIN_OBJS)

.PHONY: all run vec-report disasm clean

all: $(BINS) $(ALIASES)

# --- grouped variant objects, each with its own flags -----------------------
$(BUILD)/scalar_kernels.o: src/scalar_kernels.cpp src/kernels.h | $(BUILD)
	$(CXX) $(SCALAR_FLAGS) -c $< -o $@

$(BUILD)/autovec_kernels.o: src/autovec_kernels.cpp src/kernels.h | $(BUILD)
	$(CXX) $(AUTOVEC_FLAGS) -c $< -o $@

$(BUILD)/simd_kernels.o: src/simd_kernels.cpp src/kernels.h | $(BUILD)
	$(CXX) $(SIMD_FLAGS) -c $< -o $@

$(BUILD)/openmp_kernels.o: src/openmp_kernels.cpp src/kernels.h | $(BUILD)
	$(CXX) $(OMP_FLAGS) -c $< -o $@

# --- benchmark mains --------------------------------------------------------
$(BUILD)/%.o: src/%.cpp src/bench.h src/kernels.h | $(BUILD)
	$(CXX) $(MAIN_FLAGS) -c $< -o $@

$(BIN)/%: $(BUILD)/%.o $(GROUP_OBJS) | $(BIN)
	$(CXX) $^ -o $@ $(LINK_FLAGS)

$(BUILD) $(BIN):
	mkdir -p $@

# Alias names for the two false-sharing examples. Explicit target rules win over
# the $(BIN)/% pattern rule above, so these are symlinks, not separate links.
$(ALIASES): $(BIN)/false_sharing | $(BIN)
	ln -sf false_sharing $@

# --- running ----------------------------------------------------------------
# One BENCH line per variant on stdout; build context goes to stderr.
run: all
	@for b in $(BINS); do ./$$b; done

# --- teaching aids ----------------------------------------------------------
# First show what the autovectorizer accepted, then print -fopt-info-vec-missed
# for the loops it refused (the data-dependent Mandelbrot inner loop and the
# volatile false-sharing counters).
vec-report:
	@echo "== GCC vectorizer: autovec_kernels.cpp (-O3 -march=native) =="
	$(CXX) $(AUTOVEC_FLAGS) -fopt-info-vec-optimized \
	    -c src/autovec_kernels.cpp -o /dev/null 2>&1 || true
	@echo "== GCC -fopt-info-vec-missed: openmp_kernels.cpp =="
	$(CXX) $(OMP_FLAGS) -fopt-info-vec-missed \
	    -c src/openmp_kernels.cpp -o /dev/null 2>&1 || true

# Prove the hand-written SIMD really emits AVX2/FMA instructions.
disasm: $(BIN)/vector_add $(BIN)/dot_product
	@echo "== vector_add_avx2_fma: 256-bit vaddps =="
	@objdump -d -C --no-show-raw-insn $(BIN)/vector_add \
	    | awk '/^[0-9a-f]+ <vector_add_avx2_fma\(/{f=1} f{print} f&&/^$$/{exit}' \
	    | grep -E '<vector_add_avx2_fma|vaddps' || true
	@echo "== dot_product_avx2_fma: four independent vfmadd accumulators =="
	@objdump -d -C --no-show-raw-insn $(BIN)/dot_product \
	    | awk '/^[0-9a-f]+ <dot_product_avx2_fma\(/{f=1} f{print} f&&/^$$/{exit}' \
	    | grep -E '<dot_product_avx2_fma|vfmadd' || true

clean:
	rm -rf $(BUILD) $(BIN)
