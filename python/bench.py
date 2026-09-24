#!/usr/bin/env python3
"""Command-line benchmark harness for the Python lane.

Implements docs/BENCH_CONTRACT.md (v2.1) for the Python lane:

* ``time.perf_counter_ns`` timing, one untimed warm-up, the median over
  ``BENCH_REPS`` (default 7; the mean of the two middle samples for an even
  count), and an unconditional ``min`` plus context on stderr.
* Exactly one ``BENCH,<example>,<variant>,<n>,<ms>,<checksum>`` line per
  available variant on stdout -- stdout carries nothing else.
* ``n`` precedence: ``--n`` > ``BENCH_N`` > the per-example default; ``n <= 0``
  is rejected with a clear error.
* Element-wise checksums are computed in a separate untimed pass; reduction
  checksums are the timed kernel result, so checksums do not depend on ``REPS``.
* A module-level sink observes every reduction result so the measured work
  cannot be discarded.

Usage:
    python bench.py vector_add [--n <count>] [--reps <count>]

Run under ``taskset`` to pick the cores: ``taskset -c 2`` for a single P-core,
``taskset -c 0-11`` with ``THREADS=12`` for the multiprocessing variant.
"""

from __future__ import annotations

import os

# Pin the BLAS thread pools to one thread *before* NumPy is imported. NumPy is a
# single-thread variant per BENCH_CONTRACT.md v2.1, and forking after OpenBLAS
# has spawned threads is unsafe. This must happen before `import kernels`.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_blas_var] = "1"
del _blas_var

import argparse
import math
import platform
import shutil
import subprocess
import sys
import time
from typing import NoReturn

import kernels

DEFAULT_REPS = 7
REL_TOLERANCE = 1e-4

# Observes every reduction result. Written once more to stderr at exit, so the
# work is genuinely reachable and cannot be optimized away.
_SINK = 0.0


def consume(value: float) -> None:
    """Add a kernel result to the global sink."""
    global _SINK
    _SINK += value


def die(message: str) -> NoReturn:
    """Print a contract violation to stderr and exit with status 2."""
    print(f"bench: {message}", file=sys.stderr)
    raise SystemExit(2)


def positive_int(raw: object, what: str) -> int:
    """Parse and require a positive integer, rejecting ``n <= 0``."""
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        die(f"{what} must be a positive integer, got {raw!r}")
    if value <= 0:
        die(f"{what} must be a positive integer, got {value}")
    return value


def env_positive(name: str, fallback: int) -> int:
    """Read a positive int from the environment, else return ``fallback``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    return positive_int(raw, name)


def affinity_count() -> int:
    """Logical CPUs available to this process (1 under ``taskset -c N``).

    The worker default follows the affinity mask so a single-core run cannot
    fork a pool that timeshares one core — an oversubscription artefact, not
    parallel collapse.
    """
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, NotImplementedError):  # non-Linux / no affinity API
        return os.cpu_count() or 1
    return len(affinity) or 1


def resolve_threads() -> int:
    """Effective thread count: ``THREADS`` capped by the affinity mask."""
    available = affinity_count()
    requested = env_positive("THREADS", available)
    if requested > available:
        print(
            f"bench[python]: THREADS={requested} exceeds affinity "
            f"({available} CPUs); clamping to {available}",
            file=sys.stderr,
        )
        return available
    return requested


def gil_status() -> str:
    """Report whether this interpreter's GIL is enabled (PEP 703 build)."""
    checker = getattr(sys, "_is_gil_enabled", None)
    if checker is None:
        return "unknown (pre-3.13)"
    return "on" if checker() else "off"


def compiler_version() -> str:
    """Best-effort GCC version for the native library (context only)."""
    cc = shutil.which("gcc")
    if cc is None:
        return "n/a"
    try:
        done = subprocess.run(
            [cc, "-dumpfullversion"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "n/a"
    return done.stdout.strip() or "n/a"


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Run one Python-lane benchmark example.",
    )
    parser.add_argument(
        "example",
        choices=sorted(kernels.DEFAULTS),
        help="kernel to benchmark",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="element count (overrides BENCH_N and the per-example default)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="timed repetitions (overrides BENCH_REPS, default 7)",
    )
    return parser.parse_args(argv)


def measure(variant, reps: int) -> tuple[float, float, float]:
    """Time ``variant`` and return ``(median_ms, min_ms, checksum)``.

    The reduction checksum is the last timed result; an element-wise variant
    supplies ``checksum`` and its output is summed here, off the clock.
    """
    variant.timed()  # one untimed warm-up pass
    samples: list[float] = []
    last = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        last = variant.timed()
        stop = time.perf_counter_ns()
        if last is not None:
            consume(float(last))
        samples.append((stop - start) / 1e6)

    samples.sort()
    count = len(samples)
    if count % 2:
        median = samples[count // 2]
    else:
        median = 0.5 * (samples[count // 2 - 1] + samples[count // 2])

    if variant.checksum is not None:
        checksum = float(variant.checksum())
    elif last is not None:
        checksum = float(last)
    else:
        checksum = math.nan
    consume(checksum)
    return median, samples[0], checksum


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    example = args.example

    if args.n is not None:
        n = positive_int(args.n, "--n")
    else:
        n = env_positive("BENCH_N", kernels.DEFAULTS[example])

    if args.reps is not None:
        reps = positive_int(args.reps, "--reps")
    else:
        reps = env_positive("BENCH_REPS", DEFAULT_REPS)
    threads = resolve_threads()

    print(
        "bench[python]: example=%s n=%d reps=%d threads=%d affinity=%d "
        "python=%s numpy=%s numba=%s gil=%s gcc=%s simd=avx2+fma"
        % (
            example,
            n,
            reps,
            threads,
            affinity_count(),
            platform.python_version(),
            kernels.numpy_version(),
            kernels.NUMBA_VERSION,
            gil_status(),
            compiler_version(),
        ),
        file=sys.stderr,
    )

    def log(message: str) -> None:
        print(f"bench[python]: {message}", file=sys.stderr)

    variants = kernels.build(example, n, threads, log)
    if not variants:
        print("bench: no variants available", file=sys.stderr)
        return 1

    results: list[tuple[str, float, float, float]] = []
    for variant in variants:
        try:
            median, minimum, checksum = measure(variant, reps)
        except Exception as exc:  # a broken optional variant must not stop the run
            log(f"variant {variant.name} skipped: {type(exc).__name__}: {exc}")
            continue
        results.append((variant.name, median, minimum, checksum))

    for name, median, minimum, checksum in results:
        print(f"BENCH,{example},{name},{n},{median:.4f},{checksum:.10g}")
        sys.stdout.flush()
        print(
            "bench[%s/%s]: median=%.4f ms min=%.4f ms checksum=%.10g"
            % (example, name, median, minimum, checksum),
            file=sys.stderr,
        )

    reference_name, _, _, reference = next(
        (row for row in results if row[0] == "pure_python"),
        results[0],
    )
    ok = True
    for name, _, _, checksum in results:
        if name == reference_name:
            continue
        scale = max(abs(reference), 1e-12)
        relative = abs(checksum - reference) / scale
        if relative > REL_TOLERANCE:
            ok = False
            print(
                "bench: CORRECTNESS FAILURE %s/%s: %.10g vs %s %.10g "
                "(rel %.3e)" % (example, name, checksum, reference_name,
                                reference, relative),
                file=sys.stderr,
            )

    print("bench[python]: sink=%.10g" % _SINK, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
