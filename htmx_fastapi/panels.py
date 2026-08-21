"""Pure panel-grouping logic for the graph viewer (design D4/D5/D6).

``build_panels`` takes the lightweight :class:`Dashboard` state and returns a
list of :class:`PanelSpec` objects describing every chart/info card to render.
It is deliberately free of Reflex and ``reflex_xy`` imports so it can be unit
tested directly; :mod:`npz_visualizer.charts` reads the specs and turns each
into an XY chart.

Grouping rules (see D4/D5):

* Selected 1-D arrays of equal length form one *overlay* panel (x = index when
  ``x_key`` is ``None``). When an X array is chosen, the remaining selected
  1-D arrays whose length matches X become *Y series* on one *XY* panel.
* Selected 1-D arrays whose length differs from the chosen X are excluded from
  the XY panel and rendered as *standalone* panels, each carrying the
  ``(expected, actual)`` length mismatch that explains why.
* Each selected 2-D array gets its own *grid* panel (heatmap / contour).
* Each selected 3-D+ array gets a *slice* info card showing the first slice
  ``arr[0]`` badged as a slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from htmx_fastapi.npz_store import ArrayMeta

# 1-D mark families.
LINE_1D = ("line", "bar", "scatter", "stem")
# 2-D mark families.
GRID_2D = ("heatmap", "contour")
# 3-D+ mark families (rendered with Plotly; see main._panel_payload).
VOLUME_3D = ("surface", "scatter3d")

#: Fixed cap on the number of simultaneously visible panel slots in the main
#: grid (see :attr:`npz_visualizer.state.Dashboard` figure slots). Kept here in
#: this pure, Reflex-free module so both ``state.py`` and ``charts.py`` import
#: the same canonical value without a Reflex import cycle.
SLOT_COUNT = 8


@dataclass
class PanelSpec:
    """One renderable panel or info card.

    Attributes:
        id: Stable identifier used as the column namespace in the data method.
        kind: ``overlay`` | ``xy`` | ``standalone`` | ``grid`` | ``slice``.
        title: Human-readable panel title (array name(s)).
        series: 1-D array names that make up this panel (empty for grid/slice).
        x_key: The X array name for ``xy`` panels, else ``None``.
        plot_type: One of :data:`LINE_1D` / :data:`GRID_2D`.
        mismatched: ``(name, expected_len, actual_len)`` for excluded arrays.
        array2d_key: Key of the 2-D array for ``grid`` panels, or of the 3-D
            array for ``slice`` cards; ``None`` otherwise.
        slice_index: Axis-0 slice index for ``slice`` cards.
    """

    id: str
    kind: str
    title: str
    series: list[str] = field(default_factory=list)
    x_key: str | None = None
    plot_type: str = "line"
    mismatched: list[tuple[str, int, int]] = field(default_factory=list)
    array2d_key: str | None = None
    slice_index: int | None = None

    def to_dict(self) -> dict:
        """JSON-serializable view for the reactive UI (mismatched tuples -> lists)."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "series": list(self.series),
            "x_key": self.x_key,
            "plot_type": self.plot_type,
            "mismatched": [[a, b, c] for a, b, c in self.mismatched],
            "array2d_key": self.array2d_key,
            "slice_index": self.slice_index,
            "key": self._stable_key(),
        }

    def _stable_key(self) -> str:
        """Stable identity for rx.foreach keying across re-renders."""
        if self.kind == "overlay":
            return "|".join(self.series)
        if self.kind == "xy":
            return self.x_key or "index"
        if self.series:
            return self.series[0]
        if self.array2d_key:
            return f"{self.array2d_key}::{self.slice_index}"
        return self.id


def state_key(panel: PanelSpec) -> str:
    """Stable key into ``Dashboard.panel_types`` for a panel's plot-type override.

    - overlay -> shared empty-string key (one selector governs all 1-D overlay series)
    - xy -> the chosen X array name
    - standalone / grid / slice -> the array name (or first series / 2-D key).
    """
    if panel.kind == "overlay":
        return ""
    if panel.kind == "xy":
        return panel.x_key or ""
    if panel.array2d_key:
        return panel.array2d_key
    return panel.series[0] if panel.series else ""


def _default_plot_type(kind: str, name: str, panel_types: dict[str, str]) -> str:
    """Resolve the plot type for a panel, honouring user ``panel_types``."""
    override = panel_types.get(name)
    if kind in ("grid", "slice") and override in GRID_2D:
        return override
    if kind == "slice" and override in VOLUME_3D:
        # 3-D+ cards can switch from a flat first-slice view to a full
        # volumetric Plotly render (surface / scatter3d).
        return override
    if kind in ("overlay", "xy", "standalone") and override in LINE_1D:
        return override
    # Defaults: 2-D grids and 3-D+ first slices are heatmaps; 1-D is a line.
    return "heatmap" if kind in ("grid", "slice") else "line"


def build_panels(state) -> list[PanelSpec]:
    """Turn the open-file state into the list of panels to render.

    Reads ``state.arrays``, ``state.selected``, ``state.x_key`` and
    ``state.panel_types``. Returns an empty list when nothing is selected.
    """
    arrays = state.arrays
    selected = state.selected
    x_key = state.x_key
    panel_types = state.panel_types

    selected_arrays = [a for a in arrays if a.name in selected]
    # Non-numeric dtypes (string/bool/complex; ``stats_na`` is set from the
    # dtype at open time) cannot be plotted by the XY engine. They stay in the
    # checklist with "n/a" stats but produce no panel (spec: non-numeric
    # arrays degrade gracefully without erroring).
    plottable = [a for a in selected_arrays if not a.stats_na]
    ones = [a for a in plottable if a.ndim == 1]
    twos = [a for a in plottable if a.ndim == 2]
    threes = [a for a in plottable if a.ndim >= 3]

    panels: list[PanelSpec] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        pid = f"p{counter}"
        counter += 1
        return pid

    # ---- 1-D panels: index overlay or X/Y ------------------------------
    if x_key is None:
        # Index mode: arrays sharing a common length overlay on one chart
        # (x = index); a length outlier that cannot join gets its own
        # single-series panel (spec: Multi-array layout rules).
        by_len: dict[int, list[ArrayMeta]] = {}
        for a in ones:
            by_len.setdefault(int(a.n_elements), []).append(a)
        for length in sorted(by_len, key=lambda n: -n):
            group = by_len[length]
            names = [a.name for a in group]
            if len(group) > 1:
                panels.append(
                    PanelSpec(
                        id=next_id(),
                        kind="overlay",
                        title=f"1-D arrays (index) — length {length}",
                        series=names,
                        x_key=None,
                        plot_type=_default_plot_type("overlay", "", panel_types),
                    )
                )
            else:
                solo = group[0]
                panels.append(
                    PanelSpec(
                        id=next_id(),
                        kind="standalone",
                        title=f"{solo.name} (length {length})",
                        series=[solo.name],
                        x_key=None,
                        plot_type=_default_plot_type("standalone", solo.name, panel_types),
                    )
                )
    else:
        # X/Y mode: the chosen X drives a shared axis; only length-matching
        # arrays become Y series, the rest become standalone mismatch panels.
        x_names = x_key if isinstance(x_key, list) else [x_key]
        x_arr = next((a for a in ones if a.name in x_names), None)
        if x_arr is not None:
            x_len = int(x_arr.n_elements)
            y_series = [a.name for a in ones if a.name not in x_names and int(a.n_elements) == x_len]
            mismatched = [
                (a.name, x_len, int(a.n_elements))
                for a in ones
                if a.name not in x_names and int(a.n_elements) != x_len
            ]
            y_series_sorted = sorted(y_series)
            mismatched_sorted = sorted(mismatched, key=lambda t: t[0])
            title = "X/Y plot" + (f" vs {x_arr.name}" if len(x_names) == 1 else " vs X")
            panels.append(
                PanelSpec(
                    id=next_id(),
                    kind="xy",
                    title=title,
                    series=y_series_sorted,
                    x_key=x_names[0] if len(x_names) == 1 else x_key,
                    plot_type=_default_plot_type("xy", x_names[0], panel_types),
                    mismatched=mismatched_sorted,
                )
            )
        # Mismatched arrays become standalone single-series panels.
        for name, _exp, act in sorted(mismatched, key=lambda t: t[0]):
            panels.append(
                PanelSpec(
                    id=next_id(),
                    kind="standalone",
                    title=f"{name} (length {act})",
                    series=[name],
                    x_key=None,
                    plot_type=_default_plot_type("standalone", name, panel_types),
                    mismatched=[(x_key if isinstance(x_key, str) else x_key[0], _exp, act)],
                )
            )

    # ---- 2-D panels: one per array -------------------------------------
    for a in twos:
        panels.append(
            PanelSpec(
                id=next_id(),
                kind="grid",
                title=a.name,
                series=[],
                plot_type=_default_plot_type("grid", a.name, panel_types),
                array2d_key=a.name,
            )
        )

    # ---- 3-D+ cards: first slice of axis 0 -----------------------------
    for a in threes:
        panels.append(
            PanelSpec(
                id=next_id(),
                kind="slice",
                title=f"{a.name} (ndim {a.ndim}) — slice [0]",
                series=[],
                plot_type=_default_plot_type("slice", a.name, panel_types),
                array2d_key=a.name,
                slice_index=0,
            )
        )

    return panels
