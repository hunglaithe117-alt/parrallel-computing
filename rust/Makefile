# Convenience targets for the Rust lane.
# RUSTFLAGS enables AVX2/FMA auto-vectorization for the `autovec` variants.
# The explicit `avx2_fma` path works regardless (it uses #[target_feature]).
#
# Stdout of `make run` must contain only BENCH, lines (BENCH_CONTRACT.md v2),
# so make's own directory/recipe chatter is silenced here.
MAKEFLAGS += --no-print-directory
.SILENT:

RUSTFLAGS ?= -C target-cpu=native
export RUSTFLAGS

BINS := vector_add saxpy dot_product sum_reduce mandelbrot

.PHONY: all build run run-verbose fmt clippy clean

all: build

build:
	cargo build --release

run: build
	@for bin in $(BINS); do \
		echo "===== $$bin =====" >&2; \
		cargo run --release --quiet --bin $$bin; \
	done

run-verbose: build
	@for bin in $(BINS); do \
		echo "===== $$bin =====" >&2; \
		cargo run --release --quiet --bin $$bin -- --verbose; \
	done

fmt:
	cargo fmt --all

clippy:
	cargo clippy --release --all-targets

clean:
	cargo clean
