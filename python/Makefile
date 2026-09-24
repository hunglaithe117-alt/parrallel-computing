# Python lane of the parallel-computing lab.
#
# `setup` creates the local virtualenv and installs the pinned requirements.
# NumPy and Numba are optional in the sense that a failed install must not stop
# the lab: the harness runs whichever variants are available. `lib` compiles the
# ctypes SIMD library, and `run` executes the three examples.
#
# Contract v2 stream hygiene: bench.py keeps stdout to BENCH, lines only, so
# this Makefile silences make's own chatter.

MAKEFLAGS += --no-print-directory
.SILENT:

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CC := gcc
NATIVE := native/libsimd.so
NATIVE_CFLAGS := -O3 -mavx2 -mfma -shared -fPIC

.PHONY: setup lib run clean

# Create the venv, then install the pins. If the full set cannot be resolved
# (for example a numba wheel missing for a brand-new CPython), retry NumPy alone
# and continue: pure_python and the multiprocessing variants still run.
setup: $(PY)
	@$(PIP) install --disable-pip-version-check -r requirements.txt \
	  || (echo "setup: full requirements install failed; retrying NumPy only" >&2; \
	      $(PIP) install --disable-pip-version-check numpy==2.5.2 \
	      || echo "setup: WARNING NumPy install failed; only pure_python will run" >&2)

$(PY):
	python3 -m venv $(VENV)

lib: $(NATIVE)

$(NATIVE): native/simd.c
	$(CC) $(NATIVE_CFLAGS) -o $@ $<

run: lib
	@for ex in vector_add dot_product sum_reduce; do \
	  echo "===== $$ex =====" >&2; \
	  $(PY) bench.py $$ex; \
	done

clean:
	rm -rf $(VENV) $(NATIVE) __pycache__
