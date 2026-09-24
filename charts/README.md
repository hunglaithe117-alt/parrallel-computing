# Charts

Dependency-free chart generator for the benchmark CSVs. Python 3 **stdlib
only** — no numpy/matplotlib/pandas, no network access.

## Run

From the repository root:

```sh
python3 charts/make_charts.py
```

Exit code is 0 on success. Any warnings about skipped optional inputs go to
stderr; a short summary (row counts, files written) goes to stdout.

## Inputs (whichever exist; missing optionals are skipped gracefully)

| File | Required? | Columns |
|---|---|---|
| `results/results.csv` | yes | `language,example,variant,class,n,ms,checksum,status` |
| `results/results_t6.csv` | no (THREADS=6 snapshot) | same as `results.csv` |
| `results/scaling.csv` | no | `language,example,variant,threads,n,ms,speedup_vs_1` |

If `results/results.csv` is absent the script exits non-zero. If an optional
file is absent, the corresponding report section is replaced with a short
"omitted" note and everything else still renders.

## Outputs

| File | Contents |
|---|---|
| `docs/CHARTS.md` | Markdown with Unicode horizontal bar charts: cross-language best per kernel, within-language speedups, and thread scaling when the data exists |
| `docs/charts.html` | Self-contained HTML (inline CSS + inline SVG, no external scripts/styles/CDN) rendering the same charts |

Both outputs are deterministic for the same inputs (fixed language/kernel
ordering, sorted entries, no timestamps), so re-running is byte-stable.

## Reading the charts

- Within-language speedup = reference ms / variant ms at the same
  `(example, n)`; reference is `scalar` (Python: `pure_python`).
- Cross-language tables compare **time only** at identical `n`; checksums are
  validated within a language, never across languages.
- Bars are linear: in speedup charts longer is faster; in time charts shorter
  is faster. Units (`ms`) and `n` are shown on every chart.

## Tie / noise caveat

Per the benchmark contract: these kernels are typically **memory-bound**, and
pinned medians still vary ~10–15% run to run. **Differences under ~15% are
ties** — a row within that margin of the fastest row in its class is marked
`(tie)`, and any ordering inside the noise band must not be narrated as an
effect.
