"""Kernels and variant registry for the Python lane of the benchmark lab.

Every kernel exposes the same variants so bench.py can time them identically:

* ``pure_python``   -- lists and an explicit Python loop (the reference).
* ``numpy``         -- contiguous float32 arrays and ufuncs.
* ``numba``         -- ``@njit`` loops (optional; skipped if NumPy/Numba fail).
* ``multiprocessing`` -- a fork pool over disjoint chunks, sized to ``THREADS``.
* ``ctypes_scalar`` / ``ctypes_simd`` -- the compiled C library via ctypes.

Only one example runs per process, so the module-level arrays that the fork-pool
workers inherit through copy-on-write are shared by design. The pool is forked
before Numba's LLVM probes run, and ``bench.py`` sets ``OPENBLAS_NUM_THREADS=1``
(and friends) before importing this module: NumPy is a single-thread variant per
BENCH_CONTRACT.md v2.1, and a fork after BLAS or LLVM has spawned threads is
unsafe.
"""

from __future__ import annotations

import atexit
import ctypes
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on a broken install
    np = None

try:
    import numba as _numba

    HAVE_NUMBA = True
    NUMBA_VERSION = _numba.__version__
    _NUMBA_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    _numba = None
    HAVE_NUMBA = False
    NUMBA_VERSION = "unavailable"
    _NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

NATIVE_LIB = Path(__file__).resolve().parent / "native" / "libsimd.so"

# Per-example element counts. The pure-Python reference needs a boxed Python
# float per element (roughly 32 bytes in a list), so the defaults are sized to
# keep peak memory under ~1 GB rather than to hit a wall-clock target: at these
# counts pure_python still dominates every compiled variant by 20-100x. Anything
# larger is available via --n <count> / BENCH_N, at a proportional RAM cost.
DEFAULTS = {
    "vector_add": 8_000_000,
    "dot_product": 8_000_000,
    "sum_reduce": 16_000_000,
}


@dataclass
class Variant:
    """One named implementation of a kernel.

    ``timed`` is the only region put on the clock. For element-wise kernels it
    returns ``None`` and ``checksum`` is a separate untimed pass over the output
    buffer; for reductions it returns the scalar result, which is the checksum.
    """

    name: str
    timed: Callable[[], float | None]
    checksum: Callable[[], float] | None = None


def numpy_version() -> str:
    """The NumPy version, or ``"unavailable"`` when the import failed."""
    return np.__version__ if np is not None else "unavailable"


# ---------------------------------------------------------------------------
# Deterministic positive data
# ---------------------------------------------------------------------------
# Positive, same-magnitude values keep reduction checksums large and
# well-conditioned so the contract's 1e-4 relative tolerance is meaningful: a
# sign-cancelling sum would be dominated by rounding noise.
def _positive_array(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return (rng.random(n, dtype=np.float32) + np.float32(0.5)).astype(np.float32)


def _positive_list(n: int, seed: int) -> list[float]:
    state = (seed & 0x7FFFFFFF) or 1
    out: list[float] = []
    append = out.append
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        append(0.5 + (state & 0xFFFF) / 65535.0)
    return out


def _add_pure_reference(
    variants: list[Variant],
    a: list[float],
    b: list[float] | None,
    n: int,
    reduction: bool,
) -> None:
    """Append the ``pure_python`` reference (element-wise or reduction)."""
    if reduction:
        assert b is not None

        def timed() -> float:
            total = 0.0
            for i in range(n):
                total += a[i] * b[i]
            return total

        variants.append(Variant("pure_python", timed))
        return

    out = [0.0] * n

    def timed_elementwise() -> None:
        for i in range(n):
            out[i] = a[i] + b[i]  # type: ignore[index]

    def checksum() -> float:
        total = 0.0
        for value in out:
            total += value
        return total

    variants.append(Variant("pure_python", timed_elementwise, checksum))


# ---------------------------------------------------------------------------
# Optional numba variants
# ---------------------------------------------------------------------------
if HAVE_NUMBA:  # pragma: no cover - depends on the optional numba install

    from numba import njit

    @njit(cache=False)
    def _numba_vector_add(a, b, c, n):
        for i in range(n):
            c[i] = a[i] + b[i]

    @njit(cache=False)
    def _numba_dot(a, b, n):
        acc = 0.0
        for i in range(n):
            acc += a[i] * b[i]
        return acc

    @njit(cache=False)
    def _numba_sum(a, n):
        acc = 0.0
        for i in range(n):
            acc += a[i]
        return acc


def _numba_unavailable_reason() -> str:
    if not HAVE_NUMBA:
        return f"numba unavailable ({NUMBA_VERSION}: {_NUMBA_IMPORT_ERROR})"
    return ""


def _maybe_add_numba_vector_add(
    variants: list[Variant], a, b, n: int, log: Callable[[str], None]
) -> None:
    reason = _numba_unavailable_reason()
    if reason:
        log(f"numba vector_add skipped: {reason}")
        return
    try:
        out = np.empty(n, dtype=np.float32)
        _numba_vector_add(a[:8], b[:8], out[:8], 8)  # compile probe, untimed
    except Exception as exc:
        log(f"numba vector_add skipped: {type(exc).__name__}: {exc}")
        return
    variants.append(
        Variant(
            "numba",
            lambda: _numba_vector_add(a, b, out, n),
            lambda: float(out.sum(dtype=np.float64)),
        )
    )


def _maybe_add_numba_dot(
    variants: list[Variant], a, b, n: int, log: Callable[[str], None]
) -> None:
    reason = _numba_unavailable_reason()
    if reason:
        log(f"numba dot_product skipped: {reason}")
        return
    try:
        _numba_dot(a[:8], b[:8], 8)  # compile probe, untimed
    except Exception as exc:
        log(f"numba dot_product skipped: {type(exc).__name__}: {exc}")
        return
    variants.append(Variant("numba", lambda: float(_numba_dot(a, b, n))))


def _maybe_add_numba_sum(
    variants: list[Variant], a, n: int, log: Callable[[str], None]
) -> None:
    reason = _numba_unavailable_reason()
    if reason:
        log(f"numba sum_reduce skipped: {reason}")
        return
    try:
        _numba_sum(a[:8], 8)  # compile probe, untimed
    except Exception as exc:
        log(f"numba sum_reduce skipped: {type(exc).__name__}: {exc}")
        return
    variants.append(Variant("numba", lambda: float(_numba_sum(a, n))))


# ---------------------------------------------------------------------------
# Optional ctypes variants
# ---------------------------------------------------------------------------
def _load_lib(log: Callable[[str], None]):
    if np is None:
        log("ctypes variants skipped: numpy is required for ndpointer")
        return None
    if not NATIVE_LIB.exists():
        log(
            "ctypes variants skipped: "
            f"{NATIVE_LIB.name} missing (run `make lib`)"
        )
        return None
    try:
        lib = ctypes.CDLL(str(NATIVE_LIB))
        f32 = np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
        )
        size = ctypes.c_size_t
        dbl = ctypes.c_double

        lib.vector_add_scalar.argtypes = [f32, f32, f32, size]
        lib.vector_add_scalar.restype = None
        lib.vector_add_avx2.argtypes = [f32, f32, f32, size]
        lib.vector_add_avx2.restype = None
        lib.dot_product_scalar.argtypes = [f32, f32, size]
        lib.dot_product_scalar.restype = dbl
        lib.dot_product_avx2.argtypes = [f32, f32, size]
        lib.dot_product_avx2.restype = dbl
        lib.sum_reduce_scalar.argtypes = [f32, size]
        lib.sum_reduce_scalar.restype = dbl
        lib.sum_reduce_avx2.argtypes = [f32, size]
        lib.sum_reduce_avx2.restype = dbl
    except (OSError, AttributeError) as exc:
        log(f"ctypes variants skipped: {type(exc).__name__}: {exc}")
        return None
    return lib


# ---------------------------------------------------------------------------
# Optional multiprocessing variants
# ---------------------------------------------------------------------------
# Set before the pool is forked so workers inherit them via copy-on-write; with
# fork there is no per-call pickling of the arrays. Only one example runs per
# process, so a single set of globals is sufficient.
_MP_A = None
_MP_B = None
_MP_C = None  # shared-memory output view, vector_add only
_MP_CTX = None


def _mp_bounds(n: int, threads: int):
    bounds = np.linspace(0, n, threads + 1, dtype=np.int64)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(threads)]


def _mp_add_chunk(bounds):
    start, end = bounds
    np.add(_MP_A[start:end], _MP_B[start:end], out=_MP_C[start:end])
    return 0.0


def _mp_dot_chunk(bounds):
    start, end = bounds
    return float(np.dot(_MP_A[start:end], _MP_B[start:end]))


def _mp_sum_chunk(bounds):
    start, end = bounds
    return float(np.sum(_MP_A[start:end], dtype=np.float64))


def _make_pool(threads: int, log: Callable[[str], None]):
    global _MP_CTX
    try:
        _MP_CTX = multiprocessing.get_context("fork")
    except (ValueError, OSError) as exc:
        log(f"multiprocessing skipped: no fork context ({exc})")
        return None
    return _MP_CTX.Pool(threads)


def _register_pool_cleanup(pool, shm=None) -> None:
    def cleanup() -> None:
        try:
            pool.close()
            pool.join()
        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

    atexit.register(cleanup)


def _maybe_add_mp_vector_add(
    variants: list[Variant],
    a,
    b,
    n: int,
    threads: int,
    log: Callable[[str], None],
) -> None:
    global _MP_A, _MP_B, _MP_C
    from multiprocessing import shared_memory

    if threads <= 1:
        log(
            "multiprocessing vector_add skipped: 1 CPU in affinity mask "
            "(single-thread run; use THREADS=12 with taskset -c 0-11)"
        )
        return
    # Publish the read-only inputs and the shared output *before* forking the
    # pool: workers inherit these globals through copy-on-write.
    _MP_A, _MP_B = a, b
    shm = shared_memory.SharedMemory(create=True, size=n * a.dtype.itemsize)
    _MP_C = np.ndarray((n,), dtype=np.float32, buffer=shm.buf)
    pool = _make_pool(threads, log)
    if pool is None:
        shm.close()
        shm.unlink()
        return
    chunks = _mp_bounds(n, threads)
    pool.map(_mp_add_chunk, chunks)  # start workers, untimed

    def timed() -> None:
        pool.map(_mp_add_chunk, chunks)

    variants.append(
        Variant(
            "multiprocessing",
            timed,
            lambda: float(_MP_C.sum(dtype=np.float64)),
        )
    )
    _register_pool_cleanup(pool, shm)


def _maybe_add_mp_reduce(
    variants: list[Variant],
    chunk_fn,
    a,
    b,
    n: int,
    threads: int,
    log: Callable[[str], None],
) -> None:
    global _MP_A, _MP_B
    if threads <= 1:
        log(
            "multiprocessing reduction skipped: 1 CPU in affinity mask "
            "(single-thread run; use THREADS=12 with taskset -c 0-11)"
        )
        return
    # Publish the inputs *before* forking the pool; workers inherit them via
    # copy-on-write, so the arrays are never pickled.
    _MP_A, _MP_B = a, b
    pool = _make_pool(threads, log)
    if pool is None:
        return
    chunks = _mp_bounds(n, threads)
    pool.map(chunk_fn, chunks)  # start workers, untimed

    def timed() -> float:
        total = 0.0
        for part in pool.map(chunk_fn, chunks):
            total += part
        return total

    variants.append(Variant("multiprocessing", timed))
    _register_pool_cleanup(pool)


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------
def _build_vector_add(n: int, threads: int, log) -> list[Variant]:
    """``vector_add``: element-wise ``c = a + b``."""
    variants: list[Variant] = []
    if np is not None:
        a_arr = _positive_array(n, 0xA5A5)
        b_arr = _positive_array(n, 0x5A5A)
        a, b = a_arr.tolist(), b_arr.tolist()
    else:
        a_arr = b_arr = None
        a, b = _positive_list(n, 0xA5A5), _positive_list(n, 0x5A5A)

    _add_pure_reference(variants, a, b, n, reduction=False)

    if np is None:
        return variants

    out_np = np.empty(n, dtype=np.float32)

    def timed_numpy() -> None:
        np.add(a_arr, b_arr, out=out_np)  # ufunc returns out; timed region only

    variants.append(
        Variant(
            "numpy",
            timed_numpy,
            lambda: float(out_np.sum(dtype=np.float64)),
        )
    )
    # Fork the pool before Numba's LLVM probes: LLVM may start threads, and a
    # fork after threads exist is unsafe (the arrays already exist for COW).
    _maybe_add_mp_vector_add(variants, a_arr, b_arr, n, threads, log)
    _maybe_add_numba_vector_add(variants, a_arr, b_arr, n, log)

    lib = _load_lib(log)
    if lib is not None:
        out_scalar = np.empty(n, dtype=np.float32)
        out_simd = np.empty(n, dtype=np.float32)
        variants.append(
            Variant(
                "ctypes_scalar",
                lambda: lib.vector_add_scalar(a_arr, b_arr, out_scalar, n),
                lambda: float(out_scalar.sum(dtype=np.float64)),
            )
        )
        variants.append(
            Variant(
                "ctypes_simd",
                lambda: lib.vector_add_avx2(a_arr, b_arr, out_simd, n),
                lambda: float(out_simd.sum(dtype=np.float64)),
            )
        )
    return variants


def _build_dot_product(n: int, threads: int, log) -> list[Variant]:
    """``dot_product``: reduction ``sum(a[i] * b[i])``."""
    variants: list[Variant] = []
    if np is not None:
        a_arr = _positive_array(n, 0xD07)
        b_arr = _positive_array(n, 0xD07B)
        a, b = a_arr.tolist(), b_arr.tolist()
    else:
        a_arr = b_arr = None
        a, b = _positive_list(n, 0xD07), _positive_list(n, 0xD07B)

    _add_pure_reference(variants, a, b, n, reduction=True)

    if np is None:
        return variants

    variants.append(Variant("numpy", lambda: float(np.dot(a_arr, b_arr))))
    # Fork the pool before Numba's LLVM probes (see _build_vector_add).
    _maybe_add_mp_reduce(
        variants, _mp_dot_chunk, a_arr, b_arr, n, threads, log
    )
    _maybe_add_numba_dot(variants, a_arr, b_arr, n, log)

    lib = _load_lib(log)
    if lib is not None:
        variants.append(
            Variant(
                "ctypes_scalar",
                lambda: lib.dot_product_scalar(a_arr, b_arr, n),
            )
        )
        variants.append(
            Variant(
                "ctypes_simd", lambda: lib.dot_product_avx2(a_arr, b_arr, n)
            )
        )
    return variants


def _build_sum_reduce(n: int, threads: int, log) -> list[Variant]:
    """``sum_reduce``: reduction ``sum(a[i])``."""
    variants: list[Variant] = []
    if np is not None:
        a_arr = _positive_array(n, 0x5111)
        a = a_arr.tolist()
    else:
        a_arr = None
        a = _positive_list(n, 0x5111)

    def timed_reference() -> float:
        total = 0.0
        for value in a:
            total += value
        return total

    variants.append(Variant("pure_python", timed_reference))

    if np is None:
        return variants

    variants.append(
        Variant("numpy", lambda: float(np.sum(a_arr, dtype=np.float64)))
    )
    # Fork the pool before Numba's LLVM probes (see _build_vector_add).
    _maybe_add_mp_reduce(
        variants, _mp_sum_chunk, a_arr, None, n, threads, log
    )
    _maybe_add_numba_sum(variants, a_arr, n, log)

    lib = _load_lib(log)
    if lib is not None:
        variants.append(
            Variant("ctypes_scalar", lambda: lib.sum_reduce_scalar(a_arr, n))
        )
        variants.append(
            Variant("ctypes_simd", lambda: lib.sum_reduce_avx2(a_arr, n))
        )
    return variants


_BUILDERS = {
    "vector_add": _build_vector_add,
    "dot_product": _build_dot_product,
    "sum_reduce": _build_sum_reduce,
}


def build(
    example: str, n: int, threads: int, log: Callable[[str], None]
) -> list[Variant]:
    """Build the available variants for ``example`` with ``n`` elements.

    Allocations, data generation, pool creation, and numba compilation happen
    here, before any timing.
    """
    return _BUILDERS[example](n, threads, log)
