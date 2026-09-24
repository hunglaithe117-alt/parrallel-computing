#!/usr/bin/env python3
"""Aggregate captured ``BENCH,`` lines into results.csv and RESULTS.md.

``bench/run_all.sh`` captures the stdout ``BENCH,`` stream of every language
twice (single-thread pass and multi-thread pass).  This module is the pure
post-processing step:

* enforces the contract v2.1 merge precedence (single-thread variants only from
  the ST pass, multi-thread variants only from the MT pass);
* checks every variant against its own language's reference (``scalar``, or
  ``pure_python`` for Python) at the same ``(example, n)`` within a relative
  tolerance of ``1e-4``;
* refuses to render a cross-language time table for a kernel whose per-language
  ``n`` diverge (``MixedNError``; no ``n = mixed`` table is ever emitted);
* surfaces failed lane builds and incomplete per-pass captures in a dedicated
  "Build / capture warnings" section, so a stale-binary or partial result is
  never silently presented as fresh; and
* writes ``results/results.csv`` and renders ``docs/RESULTS.md``.

Every benchmark number in the report is derived from the captured rows; nothing
is hand-written here.  Output ordering is deterministic (fixed language/example
order, sorted variants), so re-running on the same captures is byte-identical.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Language display order and labels. Fixed so ordering never depends on the
# order files happen to be read from disk.
LANGS: list[tuple[str, str]] = [
    ("cpp", "C++"),
    ("rust", "Rust"),
    ("go", "Go"),
    ("python", "Python"),
]
LANG_ORDER = {lang: index for index, (lang, _) in enumerate(LANGS)}
LANG_LABEL = dict(LANGS)

# Contract v2.1: everything else defaults to single-thread.
MULTI_THREAD_VARIANTS = {
    "openmp",
    "pthreads",
    "std_threads",
    "rayon",
    "goroutines",
    "multiprocessing",
}

CROSS_KERNELS = ["vector_add", "saxpy", "dot_product", "sum_reduce"]
EXAMPLE_ORDER = [
    "vector_add",
    "saxpy",
    "dot_product",
    "sum_reduce",
    "mandelbrot",
    "false_sharing_adjacent",
    "false_sharing_padded",
]
EXAMPLE_INDEX = {name: index for index, name in enumerate(EXAMPLE_ORDER)}

EXCLUDED = {
    "mandelbrot": (
        "per-language `MAX_ITER`/viewport differ (contract v2.1 N2); the "
        "rendered work is not the same across languages"
    ),
    "false_sharing_adjacent": (
        "cross-language absolute times are not comparable: C++ uses a `volatile` "
        "load/store while Go must use `atomic.AddInt64` (a locked "
        "read-modify-write), so the per-increment cost differs; compare within a "
        "language"
    ),
    "false_sharing_padded": (
        "cross-language absolute times are not comparable: C++ uses a `volatile` "
        "load/store while Go must use `atomic.AddInt64` (a locked "
        "read-modify-write), so the per-increment cost differs; compare within a "
        "language"
    ),
}

TOLERANCE = 1e-4
DENOMINATOR_FLOOR = 1e-12
TIE_RATIO = 1.15

# Structural expectation of which variants each (language, example) emits, used
# only to detect a partial/crashed capture. This mirrors the variant list each
# example constructs in its source (see the per-lane READMEs); it is not a
# measured result. A capture emitting fewer variants is surfaced in the report's
# build/capture warnings.
EXPECTED_VARIANTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("cpp", "vector_add"): ("avx2_fma", "autovec", "scalar", "sse2"),
    ("cpp", "saxpy"): ("autovec", "avx2_fma", "scalar"),
    ("cpp", "dot_product"): ("autovec", "avx2_fma", "scalar"),
    ("cpp", "sum_reduce"): ("autovec", "avx2_fma", "scalar"),
    ("cpp", "mandelbrot"): ("avx2_fma", "openmp", "scalar"),
    ("cpp", "false_sharing_adjacent"): ("openmp",),
    ("cpp", "false_sharing_padded"): ("openmp",),
    ("rust", "vector_add"): ("autovec", "avx2_fma", "rayon", "scalar"),
    ("rust", "saxpy"): ("autovec", "avx2_fma", "rayon", "scalar"),
    ("rust", "dot_product"): ("autovec", "avx2_fma", "rayon", "scalar"),
    ("rust", "sum_reduce"): ("autovec", "avx2_fma", "rayon", "scalar"),
    ("rust", "mandelbrot"): ("autovec", "avx2_fma", "rayon", "scalar"),
    ("go", "vector_add"): ("goroutines", "scalar"),
    ("go", "dot_product"): ("goroutines", "scalar"),
    ("go", "sum_reduce"): ("goroutines", "scalar"),
    ("go", "false_sharing_adjacent"): ("goroutines",),
    ("go", "false_sharing_padded"): ("goroutines",),
    ("python", "vector_add"): (
        "ctypes_scalar",
        "ctypes_simd",
        "multiprocessing",
        "numba",
        "numpy",
        "pure_python",
    ),
    ("python", "dot_product"): (
        "ctypes_scalar",
        "ctypes_simd",
        "multiprocessing",
        "numba",
        "numpy",
        "pure_python",
    ),
    ("python", "sum_reduce"): (
        "ctypes_scalar",
        "ctypes_simd",
        "multiprocessing",
        "numba",
        "numpy",
        "pure_python",
    ),
}

# Variants deliberately absent from the single-thread pass: a Python pool cannot
# fork usefully on one pinned CPU, so its absence there is not a short capture.
ST_SKIPPED_VARIANTS = {("python", "multiprocessing")}

# Substrings that mark a failed build in a build_<lang>.log when no explicit
# build_status.txt manifest is available (e.g. regenerating older captures).
BUILD_FAILURE_MARKERS = (
    "make: ***",
    "error:",
    "error[",
    "fatal error:",
    "undefined reference",
    "Traceback (most recent call last)",
    "ERROR:",
    "FAILED",
)


class MixedNError(RuntimeError):
    """A cross-language kernel captured with divergent per-language ``n``."""


def variant_class(variant: str) -> str:
    """Return ``"multi-thread"`` for the contract's MT tokens, else ``"single-thread"``."""
    return "multi-thread" if variant in MULTI_THREAD_VARIANTS else "single-thread"


def class_rank(variant: str) -> int:
    """Sort key putting single-thread rows before multi-thread rows."""
    return 1 if variant_class(variant) == "multi-thread" else 0


def parse_capture(raw_dir: Path) -> list[dict]:
    """Read every ``raw_<lang>_<pass>.txt`` capture into records."""
    records: list[dict] = []
    for lang, _ in LANGS:
        for run_pass in ("st", "mt"):
            path = raw_dir / f"raw_{lang}_{run_pass}.txt"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.startswith("BENCH,"):
                    continue
                fields = line.strip().split(",")
                if len(fields) != 6:
                    print(
                        f"aggregate: malformed BENCH line in {path}: {line!r}",
                        file=sys.stderr,
                    )
                    continue
                _, example, variant, raw_n, raw_ms, raw_checksum = fields
                try:
                    n = int(raw_n)
                    ms = float(raw_ms)
                    checksum = float(raw_checksum)
                except ValueError:
                    print(
                        f"aggregate: non-numeric BENCH line in {path}: {line!r}",
                        file=sys.stderr,
                    )
                    continue
                records.append(
                    {
                        "lang": lang,
                        "pass": run_pass,
                        "example": example,
                        "variant": variant,
                        "n": n,
                        "ms": ms,
                        "checksum": checksum,
                    }
                )
    return records


def merge_by_precedence(records: list[dict]) -> tuple[dict, list[dict]]:
    """Apply the class-based merge precedence from contract v2.1.

    A single-thread row is accepted only from the ST pass, a multi-thread row
    only from the MT pass.  Rows from the wrong pass are dropped, never mixed.
    The key is ``(language, example, variant)``.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    dropped: list[dict] = []
    for record in records:
        wanted_pass = "mt" if variant_class(record["variant"]) == "multi-thread" else "st"
        if record["pass"] != wanted_pass:
            dropped.append(record)
            continue
        merged[(record["lang"], record["example"], record["variant"])] = record
    return merged, dropped


def reference_variant(lang: str) -> str:
    """The within-language correctness reference token."""
    return "pure_python" if lang == "python" else "scalar"


def check_correctness(merged: dict) -> tuple[dict, list, list]:
    """Compare each variant with its language reference at the same ``(example, n)``.

    Returns ``(status, failures, noref)`` where ``status`` maps
    ``(lang, example, variant)`` to ``"ok"``/``"FAIL"``/``"noref"``.
    """
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for record in merged.values():
        groups.setdefault((record["lang"], record["example"], record["n"]), []).append(
            record
        )

    status: dict[tuple[str, str, str], str] = {}
    failures: list[tuple] = []
    noref: list[tuple] = []
    for (lang, example, _n), rows in groups.items():
        ref_name = reference_variant(lang)
        reference = next((r for r in rows if r["variant"] == ref_name), None)
        for row in rows:
            key = (lang, example, row["variant"])
            if reference is None:
                status[key] = "noref"
                noref.append(key)
                continue
            denominator = max(abs(reference["checksum"]), DENOMINATOR_FLOOR)
            relative = abs(row["checksum"] - reference["checksum"]) / denominator
            if relative <= TOLERANCE:
                status[key] = "ok"
            else:
                status[key] = "FAIL"
                failures.append(
                    (key, relative, row["checksum"], reference["checksum"])
                )
    return status, failures, noref


def sort_key(row: dict) -> tuple:
    """Deterministic row order: language, example, class, variant."""
    return (
        LANG_ORDER[row["lang"]],
        EXAMPLE_INDEX.get(row["example"], len(EXAMPLE_ORDER)),
        class_rank(row["variant"]),
        row["variant"],
    )


def write_csv(path: Path, merged: dict, status: dict) -> None:
    """Write the merged rows with correctness status, in deterministic order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(merged.values(), key=sort_key)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["language", "example", "variant", "class", "n", "ms", "checksum", "status"]
        )
        for row in rows:
            key = (row["lang"], row["example"], row["variant"])
            writer.writerow(
                [
                    row["lang"],
                    row["example"],
                    row["variant"],
                    variant_class(row["variant"]),
                    row["n"],
                    f"{row['ms']:.4f}",
                    f"{row['checksum']:.10g}",
                    status.get(key, "noref"),
                ]
            )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_ms(value: float) -> str:
    return f"{value:,.3f}"


def ratio_cell(ms: float, best: float | None) -> str:
    """Relative-to-fastest cell with the contract's ~15% tie flag."""
    if best is None:
        return "—"
    ratio = ms / best
    if ratio <= 1.0 + 1e-9:
        return "1.00× (fastest)"
    text = f"{ratio:.2f}×"
    if ratio <= TIE_RATIO:
        text += " (tie)"
    return text


def rows_for(merged: dict, example: str) -> list[dict]:
    selected = [r for r in merged.values() if r["example"] == example]
    return sorted(selected, key=sort_key)


def render_cross_language(merged: dict) -> list[str]:
    lines: list[str] = []
    lines.append("## 1. Cross-language time")
    lines.append("")
    lines.append(
        "Time only, medians in milliseconds, at the identical `n` and "
        "`THREADS`; checksums are deliberately **not** compared across "
        "languages (input generators differ — contract v2.1). For each language "
        "the fastest captured variant in each class is shown; `tie` marks a row "
        "within 15% of the fastest row in that class."
    )
    lines.append("")

    # Coverage matrix.
    lines.append("### Kernel coverage")
    lines.append("")
    header = "| kernel | " + " | ".join(LANG_LABEL[lang] for lang, _ in LANGS) + " |"
    align = "|---|" + "|".join(":--:" for _ in LANGS) + "|"
    lines.append(header)
    lines.append(align)
    for kernel in EXAMPLE_ORDER:
        cells = []
        for lang, _ in LANGS:
            present = any(
                r["example"] == kernel and r["lang"] == lang for r in merged.values()
            )
            cells.append("✓" if present else "—")
        lines.append(f"| `{kernel}` | " + " | ".join(cells) + " |")
    lines.append("")

    for kernel in CROSS_KERNELS:
        kernel_rows = rows_for(merged, kernel)
        lines.append(f"### `{kernel}`")
        lines.append("")
        if not kernel_rows:
            lines.append("_No rows captured._")
            lines.append("")
            continue

        ns = {row["n"] for row in kernel_rows}
        # Mixed n is rejected by validate_cross_language_n() before rendering;
        # this guard keeps the renderer from ever labelling a table "mixed".
        if len(ns) != 1:
            raise MixedNError(
                f"cross-language kernel {kernel} has mixed n {sorted(ns)}; "
                "refusing to render a cross-language time table"
            )
        n_text = fmt_int(next(iter(ns)))
        implemented = [
            LANG_LABEL[lang]
            for lang, _ in LANGS
            if any(r["lang"] == lang for r in kernel_rows)
        ]
        missing = [
            LANG_LABEL[lang]
            for lang, _ in LANGS
            if not any(r["lang"] == lang for r in kernel_rows)
        ]
        lines.append(f"`n = {n_text}`")
        lines.append("")
        lines.append(
            "Implemented by: "
            + ", ".join(implemented)
            + (". Not implemented: " + ", ".join(missing) + "." if missing else ".")
        )
        lines.append("")
        lines.append(
            "| Language | best single-thread | ms | vs best ST | "
            "best multi-thread | ms | vs best MT |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        single = [
            r for r in kernel_rows if variant_class(r["variant"]) == "single-thread"
        ]
        multi = [
            r for r in kernel_rows if variant_class(r["variant"]) == "multi-thread"
        ]
        best_single = min((r["ms"] for r in single), default=None)
        best_multi = min((r["ms"] for r in multi), default=None)

        for lang, _ in LANGS:
            lang_rows = [r for r in kernel_rows if r["lang"] == lang]
            if not lang_rows:
                continue
            lang_single = [
                r for r in lang_rows if variant_class(r["variant"]) == "single-thread"
            ]
            lang_multi = [
                r for r in lang_rows if variant_class(r["variant"]) == "multi-thread"
            ]
            st_best = min(lang_single, key=lambda r: r["ms"]) if lang_single else None
            mt_best = min(lang_multi, key=lambda r: r["ms"]) if lang_multi else None
            lines.append(
                "| {label} | {st_name} | {st_ms} | {st_ratio} | "
                "{mt_name} | {mt_ms} | {mt_ratio} |".format(
                    label=LANG_LABEL[lang],
                    st_name=f"`{st_best['variant']}`" if st_best else "—",
                    st_ms=fmt_ms(st_best["ms"]) if st_best else "—",
                    st_ratio=ratio_cell(st_best["ms"], best_single) if st_best else "—",
                    mt_name=f"`{mt_best['variant']}`" if mt_best else "—",
                    mt_ms=fmt_ms(mt_best["ms"]) if mt_best else "—",
                    mt_ratio=ratio_cell(mt_best["ms"], best_multi) if mt_best else "—",
                )
            )
        lines.append("")
    return lines


def render_speedups(merged: dict) -> list[str]:
    lines: list[str] = []
    lines.append("## 2. Within-language speedups")
    lines.append("")
    lines.append(
        "Every merged variant vs that language's own reference at the same "
        "`(example, n)`: `scalar` (Python: `pure_python`). Speedup = "
        "reference ms / variant ms, so 1.00× is the reference itself."
    )
    lines.append("")

    for lang, label in LANGS:
        lang_rows = sorted(
            (r for r in merged.values() if r["lang"] == lang), key=sort_key
        )
        if not lang_rows:
            continue
        ref_name = reference_variant(lang)
        reference_by_group = {
            (r["example"], r["n"]): r["ms"]
            for r in lang_rows
            if r["variant"] == ref_name
        }
        lines.append(f"### {label} (reference: `{ref_name}`)")
        lines.append("")
        lines.append("| example | variant | class | n | ms | ref ms | speedup |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in lang_rows:
            ref_ms = reference_by_group.get((row["example"], row["n"]))
            if ref_ms is None:
                ref_text = "—"
                speed_text = "—"
            else:
                ref_text = fmt_ms(ref_ms)
                speed_text = f"{ref_ms / row['ms']:.2f}×"
            lines.append(
                f"| `{row['example']}` | `{row['variant']}` | "
                f"{variant_class(row['variant'])} | {fmt_int(row['n'])} | "
                f"{fmt_ms(row['ms'])} | {ref_text} | {speed_text} |"
            )
        lines.append("")
    return lines


def render_excluded(merged: dict) -> list[str]:
    lines: list[str] = []
    lines.append("## 3. Excluded / not cross-language-comparable")
    lines.append("")
    for example, reason in EXCLUDED.items():
        captured = sorted(
            (r for r in merged.values() if r["example"] == example), key=sort_key
        )
        if captured:
            seen: set[str] = set()
            parts = []
            for row in captured:
                if row["lang"] in seen:
                    continue
                seen.add(row["lang"])
                parts.append(f"{LANG_LABEL[row['lang']]}={fmt_int(row['n'])}")
            detail = " Captured `n`: " + ", ".join(parts) + "."
        else:
            detail = ""
        lines.append(f"- `{example}` — {reason}.{detail}")
    lines.append("")
    return lines


def render_correctness(
    merged: dict, status: dict, failures: list, noref: list
) -> tuple[list[str], bool]:
    lines: list[str] = []
    lines.append("## 4. Correctness")
    lines.append("")
    lines.append(
        "Within-language gate: every variant vs that language's own `scalar` "
        "(Python: `pure_python`) at the same `(example, n)`, relative tolerance "
        "`1e-4`, denominator `max(|reference|, 1e-12)`. `noref` marks a group "
        "with no reference row (only `false_sharing_*`, which has no `scalar`)."
    )
    lines.append("")
    lines.append("| Language | rows | ok | FAIL | noref |")
    lines.append("|---|---|---|---|---|")
    total_ok = total_fail = total_noref = 0
    for lang, label in LANGS:
        lang_keys = [
            key
            for key in status
            if key[0] == lang
        ]
        ok = sum(1 for key in lang_keys if status[key] == "ok")
        fail = sum(1 for key in lang_keys if status[key] == "FAIL")
        missing = sum(1 for key in lang_keys if status[key] == "noref")
        total_ok += ok
        total_fail += fail
        total_noref += missing
        lines.append(
            f"| {label} | {len(lang_keys)} | {ok} | {fail} | {missing} |"
        )
    lines.append(f"| **total** | **{total_ok + total_fail + total_noref}** | "
                 f"**{total_ok}** | **{total_fail}** | **{total_noref}** |")
    lines.append("")
    if failures:
        lines.append("### Failures")
        lines.append("")
        for (lang, example, variant), relative, checksum, reference in failures:
            lines.append(
                f"- `{lang}/{example}/{variant}`: checksum {checksum:.10g} vs "
                f"reference {reference:.10g} (rel {relative:.3e})"
            )
        lines.append("")
    else:
        lines.append(
            "All variants with a reference passed the within-language tolerance "
            "gate; no `FAIL` rows."
        )
        lines.append("")

    return lines, total_fail > 0


def load_skips(raw_dir: Path) -> list[tuple[str, str, str, str]]:
    """Parse ``SKIP,<lang>,<example>,<pass>,<reason>`` records."""
    path = raw_dir / "skips.txt"
    if not path.exists():
        return []
    skips = []
    for line in path.read_text().splitlines():
        if not line.startswith("SKIP,"):
            continue
        fields = line.split(",", 4)
        if len(fields) == 5:
            skips.append((fields[1], fields[2], fields[3], fields[4]))
    return skips


def expected_variants(lang: str, example: str, run_pass: str) -> set[str]:
    """The variant set an ``(lang, example, pass)`` capture should contain."""
    expected = set(EXPECTED_VARIANTS.get((lang, example), ()))
    if run_pass == "st":
        expected -= {v for (skip_lang, v) in ST_SKIPPED_VARIANTS if skip_lang == lang}
    return expected


def collect_capture_warnings(records: list[dict]) -> list[tuple]:
    """Find captures whose emitted variant set is incomplete or unexpected.

    Returns ``(lang, example, pass, missing, extra)`` tuples, sorted
    deterministically.
    """
    seen: dict[tuple[str, str, str], set[str]] = {}
    for record in records:
        seen.setdefault((record["lang"], record["pass"], record["example"]), set()).add(
            record["variant"]
        )

    warnings: list[tuple[str, str, str, list[str], list[str]]] = []
    for (lang, example), _variants in EXPECTED_VARIANTS.items():
        for run_pass in ("st", "mt"):
            expected = expected_variants(lang, example, run_pass)
            got = seen.get((lang, run_pass, example), set())
            if got != expected:
                warnings.append(
                    (
                        lang,
                        example,
                        run_pass,
                        sorted(expected - got),
                        sorted(got - expected),
                    )
                )
    warnings.sort(
        key=lambda w: (LANG_ORDER[w[0]], EXAMPLE_INDEX.get(w[1], 99), w[2])
    )
    return warnings


def validate_cross_language_n(merged: dict) -> None:
    """Refuse to rank a cross-language kernel whose per-language ``n`` diverge."""
    for kernel in CROSS_KERNELS:
        rows = [r for r in merged.values() if r["example"] == kernel]
        if not rows:
            continue
        all_ns = {r["n"] for r in rows}
        if len(all_ns) > 1:
            per_lang: dict[str, list[int]] = {}
            for row in rows:
                per_lang.setdefault(row["lang"], [])
                if row["n"] not in per_lang[row["lang"]]:
                    per_lang[row["lang"]].append(row["n"])
            detail = ", ".join(
                f"{LANG_LABEL[lang]}={sorted(ns)}"
                for lang, ns in sorted(per_lang.items(), key=lambda kv: LANG_ORDER[kv[0]])
            )
            raise MixedNError(
                f"cross-language kernel {kernel} has mixed n ({detail}); "
                "refusing to render a cross-language time table"
            )


def build_log_excerpt(text: str) -> str:
    """First failure-looking line of a build log, or ``""`` if none."""
    for line in text.splitlines():
        if any(marker in line for marker in BUILD_FAILURE_MARKERS):
            return line.strip()[:160]
    return ""


def load_build_status(raw_dir: Path) -> dict[str, tuple[str, str]]:
    """Return ``{lang: (state, evidence)}`` from the manifest, else build logs."""
    manifest = raw_dir / "build_status.txt"
    recorded: dict[str, list[str]] = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.startswith("BUILD,"):
                continue
            fields = line.split(",", 2)
            if len(fields) == 3:
                recorded.setdefault(fields[1], []).append(fields[2])

    status: dict[str, tuple[str, str]] = {}
    for lang, _ in LANGS:
        entries = recorded.get(lang, [])
        if entries:
            failed = any(e.strip().upper().startswith("FAIL") for e in entries)
            status[lang] = (
                "FAIL" if failed else "ok",
                "recorded by run_all.sh",
            )
            continue
        log = raw_dir / f"build_{lang}.log"
        if not log.exists():
            status[lang] = ("unknown", "no build_status.txt and no build log")
            continue
        excerpt = build_log_excerpt(log.read_text(errors="replace"))
        if excerpt:
            status[lang] = ("FAIL", excerpt)
        else:
            status[lang] = ("ok", f"no failure marker in build_{lang}.log")
    return status


def render_warnings(
    build_status: dict[str, tuple[str, str]],
    capture_warnings: list[tuple],
    skips: list[tuple[str, str, str, str]],
) -> list[str]:
    lines: list[str] = []
    lines.append("## Build / capture warnings")
    lines.append("")
    stale = (
        any(state == "FAIL" for state, _ in build_status.values())
        or bool(capture_warnings)
        or any(state == "unknown" for state, _ in build_status.values())
    )
    if stale:
        lines.append(
            "**WARNING: a lane build failed or a capture is incomplete — this "
            "report may contain stale-binary or partial results. Read this "
            "section before trusting the tables below.**"
        )
    else:
        lines.append(
            "No build or capture warnings: every lane's build recorded ok and "
            "every expected per-pass capture emitted its full variant set."
        )
    lines.append("")

    lines.append("### Builds")
    lines.append("")
    lines.append("| Lane | Build | Evidence |")
    lines.append("|---|---|---|")
    for lang, label in LANGS:
        state, detail = build_status.get(lang, ("unknown", "not recorded"))
        lines.append(f"| {label} | {state} | {detail} |")
    lines.append("")

    lines.append("### Capture completeness")
    lines.append("")
    if capture_warnings:
        lines.append(
            "| Language | example | pass | missing variants | unexpected variants |"
        )
        lines.append("|---|---|---|---|---|")
        for lang, example, run_pass, missing, extra in capture_warnings:
            missing_text = ", ".join(f"`{v}`" for v in missing) or "—"
            extra_text = ", ".join(f"`{v}`" for v in extra) or "—"
            lines.append(
                f"| {LANG_LABEL[lang]} | `{example}` | {run_pass} | "
                f"{missing_text} | {extra_text} |"
            )
        lines.append("")
    else:
        lines.append("Every expected per-pass capture emitted its full variant set.")
        lines.append("")

    lines.append("### Skipped runs")
    lines.append("")
    if skips:
        for lang, example, run_pass, reason in skips:
            lines.append(
                f"- {LANG_LABEL.get(lang, lang)}/{example} ({run_pass} pass): {reason}"
            )
    else:
        lines.append("None.")
    lines.append("")
    return lines


def render_markdown(
    merged: dict,
    status: dict,
    failures: list,
    noref: list,
    skips: list,
    build_status: dict[str, tuple[str, str]],
    capture_warnings: list[tuple],
    threads: int,
    reps: int,
    quick: bool,
    tag: str = "",
    pin_set: str = "0-11",
) -> tuple[str, bool]:
    raw_label = f"results/raw_{tag}/" if tag else "results/raw/"
    csv_label = f"results/results_{tag}.csv" if tag else "results/results.csv"
    pin_note = {
        "0-11": "P-cores with SMT",
        "0,2,4,6,8,10": "one logical thread per physical P-core, SMT siblings idle",
    }.get(pin_set, "P-cores")
    lines: list[str] = []
    lines.append("# Cross-language benchmark results")
    lines.append("")
    lines.append(
        "Generated by `bench/run_all.sh` (via `bench/aggregate.py`) purely from "
        f"the captured stdout `BENCH,` lines in `{raw_label}`. The merged data "
        f"is `{csv_label}`. No number in this report is hand-written."
    )
    lines.append("")
    lines.append("## Machine and methodology")
    lines.append("")
    lines.append(
        "- CPU: Intel Core i5-14400 — 6 P-cores (CPU 0-11 including SMT "
        "siblings), 4 E-cores (CPU 12-15), 1 NUMA node. AVX2 + FMA3; **no "
        "AVX-512**."
    )
    lines.append(
        "- Caches: L1d 48 KiB per P-core, L2 ~1.25 MiB per P-core, L3 20 MiB "
        "shared; 64 B cache lines."
    )
    lines.append(
        "- Governor `powersave` with turbo on: pinned medians still vary "
        "~10-15% run to run."
    )
    lines.append(
        f"- `THREADS={threads}` (`taskset -c {pin_set}`, {pin_note}); "
        "`OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, and `GOMAXPROCS` are all set "
        f"to {threads}. Single-thread rows pin `taskset -c 2` (one P-core, SMT "
        "sibling idle)."
    )
    lines.append(
        f"- `REPS={reps}` timed repetitions per row"
        + (" (`--quick` smoke settings)." if quick else ".") + " Each benchmark "
        "warms up once before timing and reports the median."
    )
    lines.append(
        "- Merge precedence (contract v2.1): single-thread variant rows are "
        "taken only from the ST pass, multi-thread rows only from the MT pass."
    )
    lines.append(
        f"- **Noise / tie rule:** differences under ~{int(round((TIE_RATIO - 1) * 100))}% "
        "are ties; a row is flagged `(tie)` when it is within that margin of "
        "the fastest row in its class for that kernel. Ordering inside the "
        "noise band is not treated as an effect."
    )
    lines.append(
        "- `scalar` is a within-language reference, not a cross-language "
        "baseline (contract v2.1 N1); the cross-language table compares time "
        "only."
    )
    lines.append(
        "- Comparison and tie decisions use the reported **median**; `min` is "
        "an advisory stability diagnostic printed to stderr and is not consumed "
        "for ranking (contract v2.1 measurement rule 7)."
    )
    lines.append("")

    lines.extend(render_warnings(build_status, capture_warnings, skips))
    lines.extend(render_cross_language(merged))
    lines.extend(render_speedups(merged))
    lines.extend(render_excluded(merged))
    correctness_lines, failed = render_correctness(merged, status, failures, noref)
    lines.extend(correctness_lines)

    return "\n".join(lines) + "\n", failed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", default="",
                        help="RUN_TAG suffix used for raw/csv/md path labels")
    parser.add_argument("--pin-set", default="0-11",
                        help="taskset CPU set used for the multi-thread pass")
    args = parser.parse_args(argv)

    records = parse_capture(args.raw_dir)
    if not records:
        print("aggregate: no BENCH, records captured", file=sys.stderr)
        return 1

    merged, dropped = merge_by_precedence(records)
    if dropped:
        print(
            f"aggregate: dropped {len(dropped)} row(s) from the wrong pass "
            "(class-based merge precedence)",
            file=sys.stderr,
        )

    try:
        validate_cross_language_n(merged)
    except MixedNError as exc:
        print(f"aggregate: {exc}", file=sys.stderr)
        print(
            "aggregate: refusing to write results.csv or RESULTS.md",
            file=sys.stderr,
        )
        return 2

    status, failures, noref = check_correctness(merged)
    write_csv(args.csv, merged, status)

    skips = load_skips(args.raw_dir)
    build_status = load_build_status(args.raw_dir)
    capture_warnings = collect_capture_warnings(records)
    markdown, failed = render_markdown(
        merged,
        status,
        failures,
        noref,
        skips,
        build_status,
        capture_warnings,
        args.threads,
        args.reps,
        args.quick,
        args.tag,
        args.pin_set,
    )
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(markdown)

    failed_builds = [lang for lang, (state, _) in build_status.items() if state == "FAIL"]
    if failed_builds or capture_warnings:
        print(
            "aggregate: WARNING "
            f"failed builds={failed_builds or 'none'} "
            f"incomplete captures={len(capture_warnings)} "
            "(see the Build / capture warnings section)",
            file=sys.stderr,
        )
    print(
        f"aggregate: {len(merged)} merged row(s); "
        f"{sum(1 for v in status.values() if v == 'ok')} ok, "
        f"{len(failures)} FAIL, "
        f"{sum(1 for v in status.values() if v == 'noref')} noref",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
