# NPZ Visualizer

<p>
  <img src="https://github.com/hitaishi2222/numpy_visualizer/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=3776AB"/>
</p>

A local, single-user web dashboard for exploring NumPy `.npz` archives — no
code required. Point it at a folder of `.npz` files, browse the named arrays
inside, pick the ones you care about (including choosing which array is **X**
and which are **Y**), and get interactive, in-browser charts.

Two implementations of the same tool live side by side:

| | [`reflex_xy/`](reflex_xy) | [`htmx_fastapi/`](htmx_fastapi) |
| --- | --- | --- |
| Stack | Reflex + [`reflex_xy`](https://reflex.dev/docs/xy/llms.txt) figure tier | FastAPI + HTMX + Tailwind + Apache ECharts |
| Charts | `xy` engine (client-side pan/zoom/hover, binary XY frames) | ECharts via CDN, JSON panel data |
| State | Reflex `State` (JSON-safe metadata only) | One in-memory `AppState` |
| Run | `uv run reflex run` (from `reflex_xy/`) → :8181 | `uv run uvicorn htmx_fastapi.main:app --port 8182` |

Both share the same feature set and the same pure core logic
(`npz_store.py`, `panels.py`).

## Features

- **Directory scanning** — enter a folder path and scan for `*.npz` files,
  listed by name with size.
- **Array enumeration** — open any `.npz` to see every named array with
  metadata (shape, dtype, element count, memory size), lazy min/max/mean/std
  statistics, and a 10-value preview for 1-D arrays.
- **Auto chart types by dimension**
  - 1-D → line chart (x = index)
  - 2-D → heatmap
  - 3-D+ → first-slice card (`arr[0]`), badged `slice [0]`; the HTMX version
    can also render the full volume in 3-D (Plotly isosurface / scatter3d)
- **Per-panel plot-type switching** — 1-D: line / bar / scatter / stem;
  2-D (and 3-D+ slices): heatmap / contour.
- **Multi-array layout** — equal-length 1-D arrays overlay as multiple series
  on one chart; each 2-D array gets its own panel; length-mismatched 1-D
  arrays get their own single-series panel.
- **X/Y assignment** — designate one selected 1-D array as X and plot the rest
  as Y series against it. Arrays whose length differs from X are excluded from
  the panel and listed beneath it with the reason (`length 150 ≠ 200`).
- **Safe by default** — archives are opened with `allow_pickle=False`, so
  object/pickled payloads are refused instead of being deserialized.

## Setup & run

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv && uv sync        # create .venv and install everything
```

**Reflex version**

```bash
cd reflex_xy
uv run reflex run         # http://localhost:8181
```

> `xy` is a pre-1.0 alpha; its exact version is pinned in `uv.lock`. If you
> rebuild the lockfile and hit adapter breakage, re-pin the known-good version
> rather than silently upgrading.

**HTMX/FastAPI version**

```bash
uv run uvicorn htmx_fastapi.main:app --port 8182   # http://localhost:8182
```

Then enter a folder path (e.g. `data/`) and click **Scan**.

## Testing

```bash
uv run pytest                                   # unit tests (npz_store, panels, charts)
uv run python scripts/e2e_npz.py                # headless end-to-end + performance checks
```

The E2E script generates sample `.npz` files, drives the scan → open → select
→ overlay → X/Y → type-switch journey, and times opening a 600k-point array
plus a 1000×1000 matrix.

## Repository layout

```
reflex_xy/               Reflex implementation
  npz_visualizer/        App package
    state.py             Dashboard state + events (scan/open/select/X-Y/type)
    npz_store.py         .npz scan, open (allow_pickle=False), cache, stats
    h5_store.py          Unified .npz + .h5/.hdf5 store (HDF5 groups flattened)
    panels.py            Pure panel-grouping logic (no Reflex)
    charts.py            The only module importing reflex_xy / xy
    ui.py                Layout (top bar, sidebar, panel grid)
    npz_visualizer.py    App entry / page registration
  rxconfig.py            Reflex config (app_name, XY plugin)

htmx_fastapi/            HTMX + FastAPI implementation (see its README)
  main.py                Routes + single-user state
  npz_store.py           Shared core (copy of the Reflex-free store)
  h5_store.py            Unified .npz + .h5/.hdf5 store (copy)
  panels.py              Shared core (copy of the grouping logic)
  templates/             Jinja2 page shell + HTMX fragments

data/                    Sample archives (.npz and .h5) for trying the apps
scripts/                 e2e_npz.py (E2E + perf), make_sample_h5.py, cdp.mjs
tests/                   pytest suite (npz_store, panels, charts)
docs/                    xy engine reference notes
openspec/                OpenSpec change history (add-npz-visualizer)
assets/                  Static assets (samples, vendored externals)
logs/                    Runtime logs (gitignored)
```

## Architecture notes (Reflex version)

Raw `ndarray`s never live in Reflex state. A server-side cache keyed by
`(path, mtime, key)` materializes arrays on first use and evicts on file
switch. Large columns ship to the browser as XY binary frames via a single
`@rxy.data` method, keeping state syncs small. Each visible panel occupies a
fixed `@rxy.figure` slot whose builder is a pure function of state; explicit
dependency declarations keep the figures re-publishing when selection / X /
file / type change.

In the HTMX version the same job is done by plain request handlers: every
mutation returns an HTML fragment that re-renders the sidebar region and
out-of-band-swaps the panel grid; chart data is fetched separately as JSON.

## Scope & safety

- **Local-only, arbitrary paths.** Single-user local tools: the sidebar
  accepts an arbitrary folder path and the server reads `.npz` files from it.
  Only `.npz` files are listed, and `allow_pickle=False` prevents pickle
  execution from untrusted archives. Do not expose either dev server on an
  untrusted network.
- **One file open at a time.** Opening a different file (or rescanning)
  clears the previous selection, X assignment, and plot-type overrides.
- **No file upload** — scanning is directory-based only.

## Roadmap

### Both versions

- [x] **HDF5 support** — open `.h5` / `.hdf5` files alongside `.npz`; nested
  groups are flattened to path-keys (`group/dataset`) and every existing
  panel type (1-D, 2-D, Plotly 3-D volumes) works unchanged.
- [ ] **More plot families** — histogram, box plot, imshow with per-channel
  RGB compositing for `(H, W, 3)` arrays, error bars / bands.
- [ ] **Slice explorer for N-D** — axis sliders (index per dimension) instead
  of the fixed `slice [0]` card, with play/animate.
- [ ] **Array math** — derived series (`a - b`, `a * 2`, rolling mean) without
  leaving the browser session.
- [ ] **Export** — download any panel as PNG/SVG or CSV of its series.
- [ ] **Session persistence** — remember dir path, open file, selection,
  X assignment, and type overrides across restarts.
- [ ] **Multiple open files** — compare arrays across archives in one grid.
- [ ] **Drag-to-zoom + crosshair sync** across panels sharing an X array.

### `reflex_xy/` specifically

- [ ] Revisit `xy` pin after its 1.0 — adopt new mark types as they land.
- [ ] Binary transfer for heatmap `z` (currently materialized server-side per
  rebuild); consider websocket streaming for very large grids.
- [ ] Panel drag-reorder / resize within the slot grid.

### `htmx_fastapi/` specifically

- [ ] Vendor Tailwind / HTMX / ECharts locally so the tool works fully
  offline.
- [ ] WebSocket channel for live-updating charts (tail a growing `.npy`).
- [ ] Per-session state (currently one global `AppState`) if multi-tab use
  ever matters.
