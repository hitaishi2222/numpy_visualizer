"""NPZ Visualizer — HTMX + Tailwind + FastAPI reimplementation.

Same feature set as the Reflex app (scan → open → select → X/Y → type
switch), but the backend is plain FastAPI and the frontend is server-rendered
HTML fragments swapped in by HTMX. Charts render client-side with Apache
ECharts; panel data is fetched as JSON from ``/data/{panel_id}``.

Run:  uv run uvicorn htmx_fastapi.main:app --reload --port 8182
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from htmx_fastapi.npz_store import (
    ArrayMeta,
    NpzError,
    evict_all,
    format_preview,
    get_array,
    get_stats,
    open_npz,
    scan_dir,
)
from htmx_fastapi.panels import GRID_2D, LINE_1D, PanelSpec, build_panels

app = FastAPI(title="NPZ Visualizer (HTMX)")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

#: Cap on points shipped per series/grid axis to keep JSON payloads sane.
MAX_POINTS = 200_000


@dataclass
class AppState:
    """Single-user, in-memory UI state (local tool: one global session)."""

    dir_path: str = ""
    scan_error: str | None = None
    files: list = field(default_factory=list)

    open_file: str | None = None
    open_error: str | None = None
    arrays: list[ArrayMeta] = field(default_factory=list)

    selected: set[str] = field(default_factory=set)
    x_key: str | None = None
    panel_types: dict[str, str] = field(default_factory=dict)
    stats: dict[str, dict] = field(default_factory=dict)


state = AppState()


# ----------------------------------------------------------------- helpers
def _reset_open_state() -> None:
    evict_all()
    state.open_file = None
    state.open_error = None
    state.arrays = []
    state.selected = set()
    state.x_key = None
    state.panel_types = {}
    state.stats = {}


def _array_by_name(name: str) -> ArrayMeta | None:
    return next((a for a in state.arrays if a.name == name), None)


def _panels() -> list[PanelSpec]:
    return build_panels(state)


def _downsample(*cols: np.ndarray) -> list[np.ndarray]:
    """Stride-downsample wide columns so the browser payload stays small."""
    n = max(len(c) for c in cols)
    if n <= MAX_POINTS:
        return list(cols)
    stride = int(np.ceil(n / MAX_POINTS))
    return [c[::stride] for c in cols]


def _panel_payload(panel: PanelSpec) -> dict:
    """JSON body for one panel consumed by the ECharts renderer."""
    if panel.kind in ("grid", "slice"):
        arr3d = get_array(_full_path(), panel.array2d_key)
        grid = arr3d[panel.slice_index] if panel.kind == "slice" else arr3d
        z = np.atleast_2d(np.asarray(grid, dtype=float))
        rows, cols = z.shape
        stride_r = max(1, int(np.ceil(rows / np.sqrt(MAX_POINTS))))
        stride_c = max(1, int(np.ceil(cols / np.sqrt(MAX_POINTS))))
        z = z[::stride_r, ::stride_c]
        return {
            "kind": panel.kind,
            "plot_type": panel.plot_type,
            "x": list(range(z.shape[1])),
            "y": list(range(z.shape[0])),
            "z": [[None if not np.isfinite(v) else v for v in row] for row in z],
        }

    full = _full_path()
    if panel.kind == "xy":
        x_arr = np.asarray(get_array(full, panel.x_key), dtype=float)
    else:
        first = get_array(full, panel.series[0])
        x_arr = np.arange(int(first.size), dtype=float)
    series = []
    for name in panel.series:
        y = np.asarray(get_array(full, name), dtype=float)
        xs, ys = _downsample(x_arr[: len(y)], y)
        series.append(
            {"name": name, "x": [None if not np.isfinite(v) else v for v in xs],
             "y": [None if not np.isfinite(v) else v for v in ys]}
        )
    return {"kind": panel.kind, "plot_type": panel.plot_type, "series": series}


def _full_path() -> str:
    return str(Path(state.dir_path) / state.open_file)


# ------------------------------------------------------------------ routes
def _render(request: Request, template: str, **ctx) -> HTMLResponse:
    ctx.setdefault("state", state)
    ctx.setdefault("LINE_1D", LINE_1D)
    ctx.setdefault("GRID_2D", GRID_2D)
    if template.endswith(("panels.html", "workspace.html", "after_scan.html")):
        ctx.setdefault("panels", _panels())
    return templates.TemplateResponse(request, template, ctx)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render(request, "index.html")


@app.post("/scan", response_class=HTMLResponse)
def scan(request: Request, dir_path: str = Form("")):
    state.dir_path = dir_path.strip()
    state.scan_error = None
    state.files = []
    try:
        state.files = scan_dir(state.dir_path)
    except NpzError as exc:
        state.scan_error = str(exc)
    _reset_open_state()
    return _render(request, "partials/after_scan.html")


@app.post("/open/{name}", response_class=HTMLResponse)
def open_file(request: Request, name: str):
    state.open_error = None
    try:
        handle = open_npz(str(Path(state.dir_path) / name))
    except NpzError as exc:
        state.open_file = name
        state.open_error = str(exc)
        state.arrays = []
        state.selected = set()
        state.x_key = None
        state.panel_types = {}
    else:
        _reset_open_state()
        state.open_file = name
        state.arrays = handle.arrays
    return _render(request, "partials/workspace.html")


@app.post("/toggle/{name}", response_class=HTMLResponse)
def toggle(request: Request, name: str):
    if name in state.selected:
        state.selected.discard(name)
    else:
        state.selected.add(name)
    if state.x_key is not None and state.x_key not in state.selected:
        arr = _array_by_name(state.x_key)
        if arr is None or arr.ndim != 1:
            state.x_key = None
    return _render(request, "partials/workspace.html")


@app.post("/select-all", response_class=HTMLResponse)
def select_all(request: Request):
    state.selected = {a.name for a in state.arrays if a.ndim == 1}
    return _render(request, "partials/workspace.html")


@app.post("/clear", response_class=HTMLResponse)
def clear(request: Request):
    state.selected = set()
    state.x_key = None
    return _render(request, "partials/workspace.html")


@app.post("/x", response_class=HTMLResponse)
def set_x(request: Request, x_key: str = Form("")):
    state.x_key = x_key or None
    return _render(request, "partials/workspace.html")


@app.post("/type/{panel_id}", response_class=HTMLResponse)
def set_type(request: Request, panel_id: str, plot_type: str = Form("")):
    panels = _panels()
    panel = next((p for p in panels if p.id == panel_id), None)
    if panel is not None and (plot_type in LINE_1D or plot_type in GRID_2D):
        key = "" if panel.kind == "overlay" else (
            panel.x_key or "" if panel.kind == "xy"
            else panel.array2d_key or (panel.series[0] if panel.series else "")
        )
        state.panel_types[key] = plot_type
    return _render(request, "partials/panels.html")


@app.get("/stats/{name}", response_class=HTMLResponse)
def stats(request: Request, name: str):
    arr = _array_by_name(name)
    if arr is None or state.open_file is None:
        return HTMLResponse("")
    try:
        s = get_stats(_full_path(), name)
        text = f"min {s.min:g} · max {s.max:g} · mean {s.mean:g} · std {s.std:g}"
    except NpzError:
        text = "stats: n/a (non-numeric)"
    state.stats[name] = {"na": arr.stats_na, "text": text}
    return _render(request, "partials/stats_line.html", name=name)


@app.get("/data/{panel_id}")
def panel_data(panel_id: str):
    panels = _panels()
    panel = next((p for p in panels if p.id == panel_id), None)
    if panel is None or state.open_file is None:
        return HTMLResponse('{"error": "no such panel"}', status_code=404)
    try:
        return _panel_payload(panel)
    except NpzError as exc:
        return {"error": str(exc)}
