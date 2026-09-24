# Go lane of the parallel-computing lab.
#
# GOAMD64=v3 makes AVX2/FMA part of the baseline, so the binary and any SIMD
# lowering target a known instruction set on this machine. The default build
# never sets GOEXPERIMENT=simd; `make simd` is the only target that pulls in the
# experimental archsimd variants.
#
# Contract v2.1: stdout must carry BENCH, lines and nothing else. Contract
# binaries go in $(BIN); `scale` prints its own table, so it is a tool and lives
# in $(TOOLSBIN) where a `bin/*` glob cannot pick it up. Build chatter is sent to
# stderr so `make run 2>/dev/null` leaves a clean stdout stream.

MAKEFLAGS += --no-print-directory
.SILENT:

GO       ?= go
GOAMD64  ?= v3
BIN      := bin
TOOLSBIN := bin-tools
SIMDBIN  := bin-simd
EXAMPLES := vector_add dot_product sum_reduce false_sharing_adjacent false_sharing_padded

.PHONY: all build run scale simd clean

all: build

build:
	@mkdir -p $(BIN) $(TOOLSBIN)
	@for ex in $(EXAMPLES); do \
		echo "  go build -o $(BIN)/$$ex ./cmd/$$ex" >&2; \
		GOAMD64=$(GOAMD64) $(GO) build -o $(BIN)/$$ex ./cmd/$$ex || exit 1; \
	done
	@echo "  go build -o $(TOOLSBIN)/scale ./cmd/scale" >&2
	@GOAMD64=$(GOAMD64) $(GO) build -o $(TOOLSBIN)/scale ./cmd/scale || exit 1

# Run every benchmark. No pinning here on purpose: use taskset yourself so the
# single-thread and multi-thread runs stay separate (see README.md).
run: build
	@for ex in $(EXAMPLES); do ./$(BIN)/$$ex; done

scale: build
	@./$(TOOLSBIN)/scale

# Experimental: simd/archsimd is unstable and GA-deferred to Go 1.28. Output
# goes to bin-simd/ so it never overwrites the default contract binaries.
simd:
	@echo "  GOEXPERIMENT=simd GOAMD64=$(GOAMD64) go build ./..." >&2
	@GOEXPERIMENT=simd GOAMD64=$(GOAMD64) $(GO) build ./...
	@mkdir -p $(SIMDBIN)
	@for ex in $(EXAMPLES); do \
		echo "  GOEXPERIMENT=simd go build -o $(SIMDBIN)/$$ex ./cmd/$$ex" >&2; \
		GOEXPERIMENT=simd GOAMD64=$(GOAMD64) $(GO) build -o $(SIMDBIN)/$$ex ./cmd/$$ex || exit 1; \
	done

clean:
	rm -rf $(BIN) $(TOOLSBIN) $(SIMDBIN)
