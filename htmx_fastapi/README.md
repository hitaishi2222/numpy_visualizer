# NPZ Visualizer — HTMX + Tailwind + FastAPI

The same `.npz` explorer as `../reflex_xy`, reimplemented without Reflex:
server-rendered HTML fragments swapped by **HTMX**, styled with **Tailwind**
(CDN), charts drawn client-side with **Apache ECharts**, backed by **FastAPI**.

## Run

```bash
uv run uvicorn htmx_fastapi.main:app --port 8182
```

Then open http://localhost:8182, enter a folder path, and Scan.
(Requires internet on first load for the Tailwind / HTMX / ECharts CDNs.)

## Features (parity with the Reflex version)

- Directory scan → file list → open `.npz` **or `.h5`/`.hdf5`**
  (`allow_pickle=False` for npz; HDF5 groups flattened to `group/dataset`
  path-keys, soft/external links skipped)
- Array cards: dim badge, shape/dtype/size, 1-D preview, lazy stats
- Selection (toggle / All / Clear) with server-driven checkbox state
- Panel grouping identical to `panels.py`: index overlay, X/Y mode with
  length-mismatch exclusion, per-2-D-array grids, 3-D+ first-slice cards
- Per-panel plot-type switching — 1-D: line/bar/scatter/stem,
  2-D: heatmap/contour (contour rendered as a smooth heat map)
- **3-D volumetric plots for N-D arrays** (Plotly): `surface` (isosurface of
  the top value shells) and `scatter3d` (translucent voxel cloud colored by
  value), each axis stride-downsampled to ≤48 samples
- Series/grid data served as JSON from `/data/{panel_id}`, stride-downsampled
  to ≤ ~200k points

## Layout

| File | Role |
| --- | --- |
| `main.py` | FastAPI routes + in-memory single-user state |
| `npz_store.py` | `.npz` scan/open/stats (copied from the Reflex app, Reflex-free) |
| `panels.py` | Pure panel-grouping logic (same as the Reflex app) |
| `templates/index.html` | Page shell + ECharts/Plotly renderer script |
| `templates/partials/*` | HTMX fragments (files, arrays, panels, stats) |

## Notes

- State is one global `AppState` — fine for the local, single-user scope;
  do not expose this server on an untrusted network.
- The checkbox state is bound server-side (`checked` rendered from state), so
  All/Clear always stay in sync — same fix as in the Reflex UI.
