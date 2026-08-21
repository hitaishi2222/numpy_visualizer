"""Chart rendering for the graph viewer (design D3).

This is the **only** module that imports ``reflex_xy`` (aliased ``rxy``) and
the ``xy`` core. There are two tiers in the adapter, and only one is used
for panels:

* **Data-bound plan tier** — ``@rxy.data`` + composed marks whose channel
  names are *literal strings* resolved at compile time. This tier cannot
  express a dynamic number of columns (channel names must be static), so it
  is **not** used for panels. It is still used by the demo ``/test-chart``
  page, which has fixed structure.

* **Figure tier (@rxy.figure)** — a state method returns a full ``xy.Chart``
  built from real numpy arrays. The adapter compiles it to a plan, mints a
  small ``FigureHandle`` token (the computed var's value), and re-publishes
  whenever the state the builder reads changes. This tier is the documented
  escape hatch for "charts whose *structure* depends on state" — exactly our
  case (a variable number of panels, each with a variable number of series
  and a user-chosen plot type). See ``reflex_xy/__init__.py`` and
  ``reflex_xy/vars.py``.

Each selectable panel gets one ``@rxy.figure`` slot (see
``npz_visualizer.state``); :func:`build_xy_figure` is the pure helper that
turns one :class:`PanelSpec` into an ``xy.Chart``. The index-mode panel
supplies an explicit ``np.arange`` element-index column because xy has no
implicit index (omitting it is the original
``ValueError: columns must be 1-D, got shape ()`` crash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import reflex_xy as rxy
import xy as _xy

from npz_visualizer import state as state_module
from npz_visualizer.panels import (
    GRID_2D,
    LINE_1D,
    SLOT_COUNT,
    PanelSpec,
    build_panels,
    state_key,
)
from npz_visualizer.h5_store import get_array

Dashboard = state_module.Dashboard

# 1-D mark families -> xy mark factory.
LINE_MARKS = {
    "line": _xy.line,
    "bar": _xy.bar,
    "scatter": _xy.scatter,
    "stem": _xy.stem,
}
# 2-D mark families -> xy mark factory.
GRID_MARKS = {
    "heatmap": _xy.heatmap,
    "contour": _xy.contour,
}
GRID_CHARTS = {"heatmap": _xy.heatmap_chart, "contour": _xy.contour_chart}


def _resolve_path(state: Dashboard) -> str | None:
    """Full path of the open file, or None if none is open / file is missing."""
    if state.open_file is None:
        return None
    full = str(Path(state.dir_path) / state.open_file)
    return full


def _index_column(panel: PanelSpec, full_path: str) -> np.ndarray:
    """Length of the first series as a 1-D element-index column (xy has no
    implicit index). Derived from a real array so length always matches the
    series; never a scalar."""
    first_key = panel.series[0] if panel.kind != "xy" else panel.x_key
    arr = get_array(full_path, first_key)
    return np.arange(int(arr.size), dtype=np.float64)


def build_xy_figure(panel: PanelSpec, full_path: str) -> Any:
    """Build a real ``xy.Chart`` for one panel from on-disk arrays.

    This is the runtime, figure-tier counterpart to the (removed) plan-tier
    ``build_chart``: marks carry real numpy arrays, so there is no channel-
    name-by-name contract to satisfy and structure can vary freely.

    Returns an ``xy.Chart`` (composed for 1-D panels, or a flat
    ``heatmap_chart``/``contour_chart`` for 2-D panels). The caller renders
    it with ``reflex_xy.chart(figure=<Dashboard figure var>)``.
    """
    kind = panel.kind
    plot_type = panel.plot_type

    if kind in ("grid", "slice"):
        arr3d = get_array(full_path, panel.array2d_key)
        grid = arr3d[panel.slice_index] if kind == "slice" else arr3d
        # xy expects 2-D z; coerce and build 1-D axis columns.
        z = np.atleast_2d(grid)
        rows, cols = z.shape
        x = np.arange(cols, dtype=np.float64)
        y = np.arange(rows, dtype=np.float64)
        mark = GRID_MARKS[plot_type](z=z, x=x, y=y, name=panel.array2d_key)
        return GRID_CHARTS[plot_type](mark)

    # ---- 1-D panels -----------------------------------------------------
    if kind == "xy":
        x_arr = get_array(full_path, panel.x_key)
    else:
        # index mode: explicit element index (never implicit, never scalar)
        x_arr = _index_column(panel, full_path)

    mark_factory = LINE_MARKS.get(plot_type, _xy.line)
    marks = []
    if kind == "xy":
        # shared x, each series is a y column
        marks.append(mark_factory(x=x_arr, y=get_array(full_path, panel.series[0]),
                                  name=panel.series[0]))
        for s in panel.series[1:]:
            marks.append(mark_factory(x=x_arr, y=get_array(full_path, s), name=s))
    else:
        # overlay / standalone: index is x, every series is a y column
        for s in panel.series:
            marks.append(mark_factory(x=x_arr, y=get_array(full_path, s), name=s))
    return _xy.chart(*marks)


def _make_figure_builder(slot: int):
    """Build a `@rxy.figure` builder for a fixed panel-slot index.

    Returns the chart for ``build_panels(self)[slot]`` or ``None`` when the
    slot is empty (the empty handle makes the XY wrapper render nothing).
    The builder is a pure function of state; the probe runs it against default
    state (no open file → ``None``) so it never touches disk at compile.
    """

    def builder(self: Dashboard):
        panels = build_panels(self)
        if slot >= len(panels):
            return None
        full_path = _resolve_path(self)
        if full_path is None or not Path(full_path).exists():
            return None
        return build_xy_figure(panels[slot], full_path)

    builder.__name__ = f"figure_{slot}"
    builder.__qualname__ = f"{__name__}.figure_{slot}"
    return builder


#: Number of compiled `@rxy.figure` slots in the panel grid. A local tool
#: rarely needs more than this many simultaneously visible panels; extra
#: panels beyond the cap are silently dropped. Bump to raise the limit.


# The figure builders read state only through the Reflex-free module helpers
# (``build_panels`` / ``_resolve_path``), so reflex's bytecode dependency scan
# sees no ``self.`` access in the builder body and would detect zero deps
# -> the figure would never rebuild when the selection changes. Declare the
# real deps explicitly so each slot is marked dirty (and re-published) on the
# right events.
_FIGURE_DEPS = ["open_file", "arrays", "selected", "x_key", "panel_types", "dir_path"]


def _register_figure_slots() -> None:
    """Attach one `@rxy.figure` computed var per panel slot to the state.

    The vars are created *after* ``Dashboard`` is defined (to avoid an import
    cycle with :mod:`npz_visualizer.charts`), so a plain ``setattr`` is not
    enough: reflex collects the state vars it will send in a delta from the
    ``vars`` / ``computed_vars`` dicts that its metaclass builds at class
    creation. A var added only via ``setattr`` never appears in ``vars``, so
    ``reflex_xy.chart(figure=Dashboard.figure_i)`` compiled into the page
    resolves to ``null`` on the client and the chart mounts blank. Replicate
    the     metaclass's mixin-var registration (see ``reflex.state``) so each
    figure slot is a first-class, delta-sent state var.

    Registering a var after class creation also misses reflex's dependency
    graph: ``BaseState.__init_subclass__`` builds ``_var_dependencies`` from
    ``cls.computed_vars`` once, at class-creation time, and reflex uses that
    table (and only that table) to mark computed vars dirty when a base var
    changes. Without the edges below, each figure evaluates exactly once —
    at hydration, against default state (no file open) — publishes an empty
    handle, and never re-runs when the user opens a file / selects arrays,
    so every chart mounts blank. Mirror the metaclass here: add each slot to
    ``_var_dependencies`` under every declared dep so state changes mark the
    slot dirty and the next delta re-evaluates (and re-publishes) it.
    """
    from reflex.vars import VarData

    full_name = Dashboard.get_full_name()
    for slot in range(SLOT_COUNT):
        builder = _make_figure_builder(slot)
        value = rxy.figure(builder, auto_deps=False, deps=_FIGURE_DEPS)
        name = f"figure_{slot}"
        fget = Dashboard._copy_fn(value.fget)
        newcv = value._replace(fget=fget, _var_data=VarData.from_state(Dashboard))
        setattr(Dashboard, name, newcv)
        Dashboard.computed_vars[name] = newcv
        Dashboard.vars[name] = newcv
        # Dirty-propagation edges (see docstring): dep var -> this slot.
        for dvar in _FIGURE_DEPS:
            Dashboard._var_dependencies.setdefault(dvar, set()).add((full_name, name))
        Dashboard._potentially_dirty_states.add(full_name)


# Attach figure slots to Dashboard on import (charts.py is imported first by
# npz_visualizer.py so the vars exist before the page compiles).
_register_figure_slots()


# ---------------------------------------------------------------------------
# UI helpers (kept here so all reflex_xy/xy imports stay in this one module):
#   type-options per panel kind, and the state key a panel writes its type
#   override under.
def type_options(kind: str) -> list[str]:
    """Plot-type choices for a panel kind (design D4)."""
    return list(LINE_1D) if kind in ("overlay", "xy", "standalone") else list(GRID_2D)


# Re-exported for state.py (which computes the per-slot reactive views).
SLOT_BUILDERS = [getattr(Dashboard, f"figure_{slot}") for slot in range(SLOT_COUNT)]


def figure_component(slot: int, height: str = "340px") -> Any:
    """Reflex component that mounts the chart for one panel ``slot``.

    Keeps every ``reflex_xy`` / ``xy`` import inside :mod:`npz_visualizer.charts`
    — :mod:`npz_visualizer.ui` only calls this and reads the slot lists.
    """
    return rxy.chart(figure=getattr(Dashboard, f"figure_{slot}"), height=height)
