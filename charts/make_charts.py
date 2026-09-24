#!/usr/bin/env python3
"""Render dependency-free charts from the benchmark CSVs.

Reads whichever of these inputs exist (stdlib only, no third-party packages):

* ``results/results.csv`` (required; columns
  ``language,example,variant,class,n,ms,checksum,status``),
* ``results/results_t6.csv`` (optional THREADS=6 snapshot, same columns),
* ``results/scaling.csv`` (optional; columns
  ``language,example,variant,threads,n,ms,speedup_vs_1``),

and writes:

* ``docs/CHARTS.md`` — Markdown with Unicode horizontal bar charts,
* ``docs/charts.html`` — self-contained HTML with inline CSS and inline SVG
  (no external scripts, stylesheets, or CDN references).

Every number shown comes from the CSVs; nothing is fabricated. Output ordering
is deterministic (fixed language/kernel order, sorted variants), so re-running
on the same inputs is byte-stable (no timestamps are emitted).
"""

from __future__ import annotations

import csv
import html as htmlmod
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / "results" / "results.csv"
RESULTS_T6_CSV = REPO_ROOT / "results" / "results_t6.csv"
SCALING_CSV = REPO_ROOT / "results" / "scaling.csv"
CHARTS_MD = REPO_ROOT / "docs" / "CHARTS.md"
CHARTS_HTML = REPO_ROOT / "docs" / "charts.html"

LANGS = [("cpp", "C++"), ("rust", "Rust"), ("go", "Go"), ("python", "Python")]
LANG_ORDER = {lang: i for i, (lang, _) in enumerate(LANGS)}
LANG_LABEL = dict(LANGS)
CROSS_KERNELS = ["vector_add", "dot_product", "sum_reduce", "saxpy"]
BAR_WIDTH = 40
TIE_MARGIN = 0.15  # Differences under ~15% are ties (contract v2.1 noise rule).
SPARK_LEVELS = "▁▂▃▄▅▆▇█"


def lang_key(lang: str) -> tuple[int, str]:
    """Sort key keeping known languages in canonical order, extras last."""
    return (LANG_ORDER.get(lang, len(LANGS)), lang)


def fmt_n(n: float | int | str) -> str:
    try:
        return f"{int(float(n)):,}"
    except (TypeError, ValueError):
        return str(n)


def fmt_ms(ms: float) -> str:
    return f"{ms:.4f} ms"


def fmt_speedup(sp: float) -> str:
    return f"{sp:.2f}x"


def bar(value: float, maximum: float, width: int = BAR_WIDTH) -> str:
    """Fixed-width Unicode bar, linear in ``value`` relative to ``maximum``."""
    if maximum <= 0 or value < 0:
        return " " * width
    filled = round(value / maximum * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "·" * (width - filled)


def spark(values: list[float]) -> str:
    """Sparkline over ``values`` using block levels (min maps to first)."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return SPARK_LEVELS[-1] * len(values)
    out = []
    for v in values:
        idx = round((v - lo) / (hi - lo) * (len(SPARK_LEVELS) - 1))
        out.append(SPARK_LEVELS[max(0, min(len(SPARK_LEVELS) - 1, idx))])
    return "".join(out)


def esc(text: str) -> str:
    return htmlmod.escape(text, quote=True)


def load_results(path: Path) -> list[dict]:
    """Load a results.csv-style file; return [] when missing, error when bad."""
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"language", "example", "variant", "class", "n", "ms"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        for rec in reader:
            try:
                rows.append(
                    {
                        "language": (rec["language"] or "").strip(),
                        "example": (rec["example"] or "").strip(),
                        "variant": (rec["variant"] or "").strip(),
                        "class": (rec.get("class") or "").strip(),
                        "n": int(float((rec["n"] or "0").strip())),
                        "ms": float((rec["ms"] or "").strip()),
                        "status": (rec.get("status") or "").strip(),
                    }
                )
            except (TypeError, ValueError):
                continue
    rows.sort(
        key=lambda r: (
            lang_key(r["language"])[0],
            r["language"],
            r["example"],
            r["variant"],
        )
    )
    return rows


def load_scaling(path: Path) -> list[dict]:
    """Load a scaling.csv-style file; raise ValueError on bad header."""
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"language", "example", "variant", "threads", "n", "ms"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        for rec in reader:
            try:
                rows.append(
                    {
                        "language": (rec["language"] or "").strip(),
                        "example": (rec["example"] or "").strip(),
                        "variant": (rec["variant"] or "").strip(),
                        "threads": int(float((rec["threads"] or "0").strip())),
                        "n": int(float((rec["n"] or "0").strip())),
                        "ms": float((rec["ms"] or "").strip()),
                        "speedup_vs_1": float((rec.get("speedup_vs_1") or "").strip()),
                    }
                )
            except (TypeError, ValueError):
                continue
    rows.sort(
        key=lambda r: (
            lang_key(r["language"])[0],
            r["language"],
            r["example"],
            r["variant"],
            r["threads"],
        )
    )
    return rows


def reference_variant(language: str) -> str:
    return "pure_python" if language == "python" else "scalar"


def kernel_n(rows: list[dict], kernel: str) -> str:
    ns = sorted({r["n"] for r in rows if r["example"] == kernel})
    if not ns:
        return "n/a"
    if len(ns) == 1:
        return fmt_n(ns[0])
    return "mixed (" + ", ".join(fmt_n(v) for v in ns) + ")"


def within_language(rows: list[dict]) -> dict[str, list[dict]]:
    """Map language -> speedup entries vs its own reference, sorted desc."""
    ref_ms: dict[tuple[str, str, int], float] = {}
    for r in rows:
        if r["variant"] == reference_variant(r["language"]):
            key = (r["language"], r["example"], r["n"])
            if key not in ref_ms:
                ref_ms[key] = r["ms"]
    out: dict[str, list[dict]] = {}
    for r in rows:
        key = (r["language"], r["example"], r["n"])
        ref = ref_ms.get(key)
        if ref is None or ref <= 0 or r["ms"] <= 0:
            continue
        out.setdefault(r["language"], []).append(
            {
                "example": r["example"],
                "variant": r["variant"],
                "class": r["class"],
                "n": r["n"],
                "ms": r["ms"],
                "ref_ms": ref,
                "speedup": ref / r["ms"],
                "is_ref": r["variant"] == reference_variant(r["language"]),
            }
        )
    for lang in out:
        out[lang].sort(
            key=lambda e: (-e["speedup"], e["example"], e["variant"])
        )
    return out


def best_per_class(
    rows: list[dict], kernel: str, class_name: str
) -> list[dict]:
    """Best (min ms) row per language for one kernel and thread class."""
    best: dict[str, dict] = {}
    for r in rows:
        if r["example"] != kernel or r["class"] != class_name:
            continue
        if r["ms"] <= 0:
            continue
        if r["language"] not in best or r["ms"] < best[r["language"]]["ms"]:
            best[r["language"]] = r
    ordered = sorted(best.values(), key=lambda r: (r["ms"], r["language"]))
    if not ordered:
        return []
    fastest = ordered[0]["ms"]
    out = []
    for r in ordered:
        ratio = r["ms"] / fastest if fastest > 0 else 1.0
        out.append(
            {
                "language": r["language"],
                "variant": r["variant"],
                "ms": r["ms"],
                "n": r["n"],
                "ratio": ratio,
                "tie": ratio - 1.0 < TIE_MARGIN,
            }
        )
    return out


def md_cross_section(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    lines.append("## 1. Cross-language best per kernel (time only)")
    lines.append("")
    lines.append(
        "For each kernel and thread class, the fastest captured variant per "
        "language is shown. Bars are linear in milliseconds, so a *shorter* "
        "bar is faster. `(tie)` marks a row within 15% of the fastest row "
        "in its class."
    )
    lines.append("")
    any_kernel = False
    for kernel in CROSS_KERNELS:
        st = best_per_class(rows, kernel, "single-thread")
        mt = best_per_class(rows, kernel, "multi-thread")
        if not st and not mt:
            continue
        any_kernel = True
        lines.append(f"### `{kernel}` (n = {kernel_n(rows, kernel)})")
        lines.append("")
        for title, group in (
            ("best single-thread (shorter = faster)", st),
            ("best multi-thread (shorter = faster)", mt),
        ):
            if not group:
                lines.append(f"_No {title.split()[1]} rows captured._")
                lines.append("")
                continue
            lines.append(f"_{title}:_")
            lines.append("")
            lines.append("```")
            max_ms = max(g["ms"] for g in group)
            label_w = max(len(f"{LANG_LABEL.get(g['language'], g['language'])}") for g in group)
            for g in group:
                label = LANG_LABEL.get(g["language"], g["language"])
                tag = " (tie)" if g["tie"] and g["ratio"] > 1.0 else ""
                if g["ratio"] == 1.0:
                    tag = " (fastest)"
                lines.append(
                    f"{label:<{label_w}} {bar(g['ms'], max_ms)} "
                    f"{g['ms']:>10.4f} ms  {g['variant']:<14} "
                    f"{g['ratio']:.2f}x{tag}"
                )
            lines.append("```")
            lines.append("")
    if not any_kernel:
        lines.append("_No cross-language kernels captured._")
        lines.append("")
    return lines


def md_within_section(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    lines.append("## 2. Within-language speedups (reference = `scalar`, Python `pure_python`)")
    lines.append("")
    lines.append(
        "Speedup = reference ms / variant ms at the same `(example, n)`, so "
        "1.00x is the reference itself. Entries are sorted by speedup, "
        "descending. Bars are linear in speedup, so a *longer* bar is faster "
        "relative to its own reference."
    )
    lines.append("")
    groups = within_language(rows)
    for lang, _ in sorted(LANGS, key=lambda t: lang_key(t[0])):
        entries = groups.get(lang, [])
        if not entries:
            continue
        ref = reference_variant(lang)
        lines.append(f"### {LANG_LABEL[lang]} (reference `{ref}`)")
        lines.append("")
        lines.append("```")
        max_sp = max(e["speedup"] for e in entries)
        label_w = max(len(f"{e['example']}/{e['variant']}") for e in entries)
        for e in entries:
            label = f"{e['example']}/{e['variant']}"
            lines.append(
                f"{label:<{label_w}} {bar(e['speedup'], max_sp)} "
                f"{e['speedup']:>7.2f}x  {e['ms']:>10.4f} ms  "
                f"(ref {e['ref_ms']:.4f} ms, n={fmt_n(e['n'])}, {e['class']})"
            )
        lines.append("```")
        lines.append("")
    # Any extra languages not in the canonical list.
    for lang in sorted(groups, key=lang_key):
        if lang in LANG_LABEL:
            continue
        entries = groups[lang]
        ref = reference_variant(lang)
        lines.append(f"### {lang} (reference `{ref}`)")
        lines.append("")
        lines.append("```")
        max_sp = max(e["speedup"] for e in entries)
        label_w = max(len(f"{e['example']}/{e['variant']}") for e in entries)
        for e in entries:
            label = f"{e['example']}/{e['variant']}"
            lines.append(
                f"{label:<{label_w}} {bar(e['speedup'], max_sp)} "
                f"{e['speedup']:>7.2f}x  {e['ms']:>10.4f} ms  "
                f"(ref {e['ref_ms']:.4f} ms, n={fmt_n(e['n'])}, {e['class']})"
            )
        lines.append("```")
        lines.append("")
    return lines


def md_scaling_section(scaling: list[dict] | None) -> list[str]:
    lines: list[str] = []
    lines.append("## 3. Thread scaling")
    lines.append("")
    if not scaling:
        lines.append(
            "_Omitted: `results/scaling.csv` was not present when this file "
            "was generated. Re-run `python3 charts/make_charts.py` after "
            "producing it to include per-thread bars._"
        )
        lines.append("")
        return lines
    lines.append(
        "Per `(language, example, variant)` across threads 1..12. `ms` is the "
        "median milliseconds at that thread count; `speedup_vs_1` comes from "
        "the CSV. The sparkline tracks `speedup_vs_1` (higher is better)."
    )
    lines.append("")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in scaling:
        grouped.setdefault(
            (r["language"], r["example"], r["variant"]), []
        ).append(r)
    for key in sorted(grouped, key=lambda k: (lang_key(k[0])[0], k)):
        pts = sorted(grouped[key], key=lambda r: r["threads"])
        lang, example, variant = key
        n_txt = (
            fmt_n(pts[0]["n"])
            if len({p["n"] for p in pts}) == 1
            else "mixed"
        )
        lines.append(f"### {lang} `{example}` / `{variant}` (n = {n_txt})")
        lines.append("")
        lines.append("```")
        max_sp = max(p["speedup_vs_1"] for p in pts) or 1.0
        for p in pts:
            lines.append(
                f"t={p['threads']:>2} {bar(p['speedup_vs_1'], max_sp)} "
                f"{p['speedup_vs_1']:>6.2f}x  {p['ms']:>10.4f} ms"
            )
        lines.append(f"spark(speedup_vs_1): {spark([p['speedup_vs_1'] for p in pts])}")
        lines.append("```")
        lines.append("")
    return lines


def md_t6_section(t6_rows: list[dict] | None) -> list[str]:
    lines: list[str] = []
    if not t6_rows:
        return lines
    lines.append("## 4. THREADS=6 snapshot (`results/results_t6.csv`)")
    lines.append("")
    lines.append(
        "Same cross-language best-per-class view as section 1, but for the "
        "optional THREADS=6 run."
    )
    lines.append("")
    for kernel in CROSS_KERNELS:
        st = best_per_class(t6_rows, kernel, "single-thread")
        mt = best_per_class(t6_rows, kernel, "multi-thread")
        if not st and not mt:
            continue
        lines.append(f"### `{kernel}` (n = {kernel_n(t6_rows, kernel)})")
        lines.append("")
        for title, group in (("single-thread", st), ("multi-thread", mt)):
            if not group:
                continue
            lines.append(f"_{title}:_")
            lines.append("")
            lines.append("```")
            max_ms = max(g["ms"] for g in group)
            label_w = max(len(g["language"]) for g in group)
            for g in group:
                tag = " (tie)" if g["tie"] and g["ratio"] > 1.0 else ""
                if g["ratio"] == 1.0:
                    tag = " (fastest)"
                lines.append(
                    f"{g['language']:<{label_w}} {bar(g['ms'], max_ms)} "
                    f"{g['ms']:>10.4f} ms  {g['variant']:<14} "
                    f"{g['ratio']:.2f}x{tag}"
                )
            lines.append("```")
            lines.append("")
    return lines


def render_markdown(
    rows: list[dict],
    t6_rows: list[dict] | None,
    scaling: list[dict] | None,
    t6_present: bool,
    scaling_present: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Benchmark charts")
    lines.append("")
    lines.append(
        "Generated by `python3 charts/make_charts.py` (Python 3 stdlib only: "
        "no numpy/matplotlib/pandas, no network) purely from the CSVs below. "
        "No number here is hand-written."
    )
    lines.append("")
    lines.append("## 0. Data sources and context")
    lines.append("")
    lines.append(f"- `results/results.csv`: {len(rows)} rows (required).")
    if t6_present:
        lines.append(
            f"- `results/results_t6.csv`: {len(t6_rows or [])} rows "
            "(optional THREADS=6 run)."
        )
    else:
        lines.append("- `results/results_t6.csv`: absent (optional; skipped).")
    if scaling_present:
        lines.append(
            f"- `results/scaling.csv`: {len(scaling or [])} rows (optional)."
        )
    else:
        lines.append(
            "- `results/scaling.csv`: absent (optional; scaling section omitted)."
        )
    lines.append(
        "- `n` (elements) is shown per kernel/chart and comes from the CSV. "
        "The `class` column (`single-thread`/`multi-thread`) marks the thread "
        "class of each row; the multi-thread row pass uses the run's THREADS "
        "setting, which is recorded in `docs/RESULTS.md`, not in "
        "`results.csv`, so no thread count is invented here."
    )
    lines.append("")
    lines.append(
        "> Caveats (from the benchmark contract): the element-wise and "
        "reduction kernels here are typically **memory-bound**, so SIMD or "
        "threading speedups are limited by bandwidth, not arithmetic. "
        "Differences **under ~15% are ties** (noise band: pinned medians "
        "still vary ~10-15% run to run); never narrate an ordering inside "
        "the band as an effect. Cross-language comparison is **time-only**: "
        "input generators differ by design, so checksums are validated "
        "within a language, never across languages."
    )
    lines.append("")
    lines.extend(md_cross_section(rows))
    lines.extend(md_within_section(rows))
    lines.extend(md_scaling_section(scaling))
    t6_lines = md_t6_section(t6_rows)
    if t6_lines:
        lines.extend(t6_lines)
    lines.append("## Sources")
    lines.append("")
    lines.append(
        "- Inputs: `results/results.csv`"
        + ("`, `results/results_t6.csv`" if t6_present else "")
        + ("`, `results/scaling.csv`" if scaling_present else "")
        + "."
    )
    lines.append("- Method: reference = `scalar` (Python: `pure_python`); "
                 "speedup = reference ms / variant ms.")
    lines.append("- Tie/noise rule: within 15% of the fastest row in its "
                 "class is a tie.")
    lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def svg_hbars(
    items: Sequence[tuple[str, float, str]],
    max_value: float,
    unit_note: str = "",
    width: int = 680,
    row_h: int = 26,
    label_w: int = 170,
    value_w: int = 210,
) -> str:
    """Inline SVG horizontal bars. Values scale linearly to ``max_value``."""
    pad = 8
    bar_area = width - label_w - value_w - pad * 3
    height = max(row_h, row_h * len(items) + pad * 2)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'style="max-width:100%;height:auto;display:block">'
    ]
    if unit_note:
        parts.append(f"<title>{esc(unit_note)}</title>")
    for i, (label, value, annotation) in enumerate(items):
        y = pad + i * row_h
        frac = (value / max_value) if max_value > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        bw = round(frac * bar_area)
        parts.append(
            f'<text x="{pad}" y="{y + 16}" font-size="12" '
            f'font-family="monospace">{esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 2}" width="{bw}" height="16" '
            f'fill="#2f6feb" rx="2"/>'
        )
        parts.append(
            f'<text x="{label_w + bar_area + pad}" y="{y + 16}" '
            f'font-size="12" font-family="monospace">{esc(annotation)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(
    rows: list[dict],
    t6_rows: list[dict] | None,
    scaling: list[dict] | None,
    t6_present: bool,
    scaling_present: bool,
) -> str:
    css = (
        "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;"
        "padding:0 1rem;color:#111;background:#fff}"
        "h1,h2,h3{line-height:1.3}code{background:#f2f2f2;padding:0 .3em;"
        "border-radius:4px}pre{background:#f6f8fa;padding:1rem;overflow:auto;"
        "border-radius:8px}.note{background:#fff8e1;border:1px solid #e6c200;"
        "padding:1rem;border-radius:8px}section{margin:2rem 0}svg text{fill:#111}"
        "table{border-collapse:collapse}footer{color:#555;font-size:.9rem}"
    )
    p: list[str] = []
    p.append("<!DOCTYPE html>")
    p.append('<html lang="en"><head><meta charset="utf-8">')
    p.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    p.append(f"<title>Benchmark charts</title><style>{css}</style></head><body>")
    p.append("<h1>Benchmark charts</h1>")
    p.append(
        "<p>Generated by <code>python3 charts/make_charts.py</code> "
        "(Python 3 stdlib only) purely from the CSVs. No number here is "
        "hand-written.</p>"
    )
    p.append('<div class="note"><strong>Caveats (from the benchmark contract):'
             "</strong> the element-wise and reduction kernels here are "
             "typically <strong>memory-bound</strong>, so SIMD or threading "
             "speedups are limited by bandwidth. Differences "
             "<strong>under ~15% are ties</strong> (noise band); never narrate "
             "an ordering inside the band as an effect. Cross-language "
             "comparison is <strong>time-only</strong>: input generators "
             "differ by design, so checksums are validated within a language, "
             "never across languages.</div>")
    p.append(
        f"<p>Sources: <code>results/results.csv</code> ({len(rows)} rows)"
        + (f"; <code>results/results_t6.csv</code> ({len(t6_rows or [])} rows)"
           if t6_present else "; <code>results/results_t6.csv</code> absent")
        + (f"; <code>results/scaling.csv</code> ({len(scaling or [])} rows)"
           if scaling_present else "; <code>results/scaling.csv</code> absent")
        + ". <code>n</code> is shown per chart from the CSV; thread class "
        "comes from the CSV <code>class</code> column.</p>"
    )

    p.append("<section><h2>Cross-language best per kernel (time only)</h2>")
    p.append("<p>Shorter bar = faster. <code>(tie)</code> = within 15% of "
             "the fastest row in its class.</p>")
    for kernel in CROSS_KERNELS:
        st = best_per_class(rows, kernel, "single-thread")
        mt = best_per_class(rows, kernel, "multi-thread")
        if not st and not mt:
            continue
        p.append(f"<h3><code>{esc(kernel)}</code> (n = {esc(kernel_n(rows, kernel))})</h3>")
        for title, group in (
            ("Best single-thread", st),
            ("Best multi-thread", mt),
        ):
            if not group:
                continue
            p.append(f"<h4>{esc(title)}</h4>")
            max_ms = max(float(g["ms"]) for g in group)
            items: list[tuple[str, float, str]] = [
                (
                    str(LANG_LABEL.get(str(g["language"]), str(g["language"]))),
                    float(g["ms"]),
                    f"{g['ms']:.4f} ms  {g['variant']}  {g['ratio']:.2f}x"
                    + (" (tie)" if g["tie"] and g["ratio"] > 1.0 else ""
                       if g["ratio"] > 1.0 else " (fastest)"),
                )
                for g in group
            ]
            p.append(svg_hbars(items, max_ms, f"{kernel} {title} milliseconds"))
    p.append("</section>")

    p.append("<section><h2>Within-language speedups</h2>")
    p.append("<p>Reference: <code>scalar</code> (Python: "
             "<code>pure_python</code>). Speedup = reference ms / variant ms. "
             "Longer bar = faster vs its own reference.</p>")
    groups = within_language(rows)
    for lang, _ in sorted(LANGS, key=lambda t: lang_key(t[0])):
        entries = groups.get(lang, [])
        if not entries:
            continue
        ref = reference_variant(lang)
        p.append(f"<h3>{esc(LANG_LABEL[lang])} (reference <code>{esc(ref)}</code>)</h3>")
        max_sp = max(e["speedup"] for e in entries)
        items = [
            (
                f"{e['example']}/{e['variant']}",
                e["speedup"],
                f"{e['speedup']:.2f}x  {e['ms']:.4f} ms (n={fmt_n(e['n'])})",
            )
            for e in entries
        ]
        p.append(svg_hbars(items, max_sp, f"{lang} speedup vs {ref}"))
    for lang in sorted(groups, key=lang_key):
        if lang in LANG_LABEL:
            continue
        entries = groups[lang]
        ref = reference_variant(lang)
        p.append(f"<h3>{esc(lang)} (reference <code>{esc(ref)}</code>)</h3>")
        max_sp = max(e["speedup"] for e in entries)
        items = [
            (
                f"{e['example']}/{e['variant']}",
                e["speedup"],
                f"{e['speedup']:.2f}x  {e['ms']:.4f} ms (n={fmt_n(e['n'])})",
            )
            for e in entries
        ]
        p.append(svg_hbars(items, max_sp, f"{lang} speedup vs {ref}"))
    p.append("</section>")

    p.append("<section><h2>Thread scaling</h2>")
    if not scaling:
        p.append("<p><em>Omitted: <code>results/scaling.csv</code> was not "
                 "present when this file was generated.</em></p>")
    else:
        p.append("<p>Bars show <code>speedup_vs_1</code> from the CSV "
                 "(longer = better).</p>")
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for r in scaling:
            grouped.setdefault(
                (r["language"], r["example"], r["variant"]), []
            ).append(r)
        for key in sorted(grouped, key=lambda k: (lang_key(k[0])[0], k)):
            pts = sorted(grouped[key], key=lambda r: r["threads"])
            lang, example, variant = key
            n_txt = (
                fmt_n(pts[0]["n"])
                if len({q["n"] for q in pts}) == 1
                else "mixed"
            )
            p.append(
                f"<h3>{esc(lang)} <code>{esc(example)}</code> / "
                f"<code>{esc(variant)}</code> (n = {esc(n_txt)})</h3>"
            )
            max_sp = max(q["speedup_vs_1"] for q in pts) or 1.0
            items = [
                (
                    f"t={q['threads']}",
                    q["speedup_vs_1"],
                    f"{q['speedup_vs_1']:.2f}x  {q['ms']:.4f} ms",
                )
                for q in pts
            ]
            p.append(svg_hbars(items, max_sp, f"{lang} {example} {variant} scaling"))
    p.append("</section>")

    if t6_rows:
        p.append("<section><h2>THREADS=6 snapshot</h2>")
        for kernel in CROSS_KERNELS:
            st = best_per_class(t6_rows, kernel, "single-thread")
            mt = best_per_class(t6_rows, kernel, "multi-thread")
            if not st and not mt:
                continue
            p.append(
                f"<h3><code>{esc(kernel)}</code> "
                f"(n = {esc(kernel_n(t6_rows, kernel))})</h3>"
            )
            for title, group in (("Single-thread", st), ("Multi-thread", mt)):
                if not group:
                    continue
                p.append(f"<h4>{esc(title)}</h4>")
                max_ms = max(g["ms"] for g in group)
                items = [
                    (
                        g["language"],
                        g["ms"],
                        f"{g['ms']:.4f} ms  {g['variant']}  {g['ratio']:.2f}x",
                    )
                    for g in group
                ]
                p.append(svg_hbars(items, max_ms, f"{kernel} {title} T6"))
        p.append("</section>")

    p.append("<footer><p>Method: reference = <code>scalar</code> (Python: "
             "<code>pure_python</code>); speedup = reference ms / variant ms; "
             "ties = within 15% of fastest in class.</p></footer>")
    p.append("</body></html>")
    return "\n".join(p) + "\n"


def main() -> int:
    if not RESULTS_CSV.is_file():
        print(f"error: required input missing: {RESULTS_CSV}", file=sys.stderr)
        return 1
    try:
        rows = load_results(RESULTS_CSV)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    t6_rows: list[dict] | None = None
    t6_present = RESULTS_T6_CSV.is_file()
    if t6_present:
        try:
            t6_rows = load_results(RESULTS_T6_CSV)
        except ValueError as exc:
            print(f"warning: skipping {RESULTS_T6_CSV}: {exc}", file=sys.stderr)
            t6_rows = None
            t6_present = False
    scaling: list[dict] | None = None
    scaling_present = SCALING_CSV.is_file()
    if scaling_present:
        try:
            scaling = load_scaling(SCALING_CSV)
        except ValueError as exc:
            print(f"warning: skipping {SCALING_CSV}: {exc}", file=sys.stderr)
            scaling = None
            scaling_present = False
    md = render_markdown(rows, t6_rows, scaling, t6_present, scaling_present)
    html_doc = render_html(rows, t6_rows, scaling, t6_present, scaling_present)
    CHARTS_MD.parent.mkdir(parents=True, exist_ok=True)
    CHARTS_MD.write_text(md, encoding="utf-8", newline="\n")
    CHARTS_HTML.write_text(html_doc, encoding="utf-8", newline="\n")
    print(f"rows: {len(rows)} from {RESULTS_CSV}")
    print(f"t6: {'absent' if not t6_present else f'{len(t6_rows or [])} rows'}")
    print(
        "scaling: "
        + ("absent" if not scaling_present else f"{len(scaling or [])} rows")
    )
    print(f"wrote {CHARTS_MD}")
    print(f"wrote {CHARTS_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
