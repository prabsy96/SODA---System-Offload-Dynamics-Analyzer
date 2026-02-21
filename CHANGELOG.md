# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Empirical KV cache measurement** — `_compute_kv_cache_bytes()` helper in `__init__.py` reads `past_key_values` from the last profiled `generate()` call (via `return_dict_in_generate=True`) and sums actual K/V tensor bytes. Handles `DynamicCache`/`StaticCache` (HF ≥ 4.36) and legacy tuple-of-tuples. Result stored as `memory_metrics.kv_cache_mb` in `report.json` and displayed in the Memory section of the console/HTML summary. Automatically captures GQA, quantized KV, sliding-window, and MLA layouts without model-specific code. See `assumption.md` §10.
- **Self-contained comparative HTML report** — heatmap PNGs are now embedded as base64 data URIs in `comparative_report.html`. The report is fully self-contained and opens correctly from any path, browser, or viewer without requiring co-located image files.

### Fixed
- **GPU name lookup in sweep report** — `get_gpu_name()` now reads from `metadata.config.gpu_name` (the correct SODA schema location) before falling back to legacy keys. Previously always returned `"gpu"`, causing H100/H200 suffixes to be absent from PNG slugs and section headers.

### Changed
- **Removed status labels from all report outputs** — OK/WARN/CRIT health labels removed from: the Rich console GPU table (Utilization row), the per-run `report.html` KV table, and the comparative HTML run table. The "Status" column (OK/OOM badge) is removed from the comparative HTML table. OOM cells still show "OOM" in red via individual data cells. CSS classes `.ok`, `.warn`, `.crit`, `td.status` removed from `_HTML_CSS`.

## [0.1.0] - 2025-09-03
### Added
- Initial public release of SODA.
- CLI interface with support for model loading, tracing, and fusion analysis.
- Basic metrics: runtime, launch overhead, idle time, kernel frequency.

### Changed
- Light polish to `main.py` and `util.py` (docstrings, logging, type hints).

### Fixed
- CLI validation for unsupported precision on CPU.