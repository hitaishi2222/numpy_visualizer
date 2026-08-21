"""Unit tests for npz_visualizer.panels.build_panels (design D4/D5/D6).

``build_panels`` is a pure function of the lightweight dashboard state, so we
feed it a small stand-in object built from :class:`ArrayMeta` records — no
Reflex, no ``reflex_xy``, no files on disk.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from npz_visualizer.npz_store import ArrayMeta
from npz_visualizer.panels import build_panels


def _arr(name: str, shape, dtype="float64") -> ArrayMeta:
    """Minimal ArrayMeta for build_panels tests.

    ``stats_na`` is derived from the dtype exactly as production does in
    :func:`npz_visualizer.npz_store.open_npz` (non-numeric => True).
    """
    dt = np.dtype(dtype)
    ndim = len(shape)
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    numeric = np.issubdtype(dt, np.integer) or np.issubdtype(dt, np.floating)
    return ArrayMeta(
        name=name,
        shape=list(shape),
        dtype=str(dt),
        ndim=ndim,
        n_elements=n,
        nbytes=n * dt.itemsize,
        stats_na=not numeric,
    )


def make_state(arrays, selected=None, x_key=None, panel_types=None):
    return SimpleNamespace(
        arrays=arrays,
        selected=selected if selected is not None else {a.name for a in arrays},
        x_key=x_key,
        panel_types=panel_types if panel_types is not None else {},
    )


def kinds(panels):
    return [p.kind for p in panels]


# --------------------------------------------------------------------- index


class TestIndexMode:
    def test_no_selection_returns_no_panels(self):
        s = make_state([_arr("x", (10,))], selected=set())
        assert build_panels(s) == []

    def test_equal_length_ones_overlay_single_panel(self):
        s = make_state([_arr("a", (100,)), _arr("b", (100,)), _arr("c", (100,))])
        panels = build_panels(s)
        assert len(panels) == 1
        p = panels[0]
        assert p.kind == "overlay"
        assert p.series == ["a", "b", "c"]
        assert p.x_key is None
        assert p.plot_type == "line"

    def test_mismatched_one_gets_own_standalone_panel(self):
        s = make_state([_arr("a", (100,)), _arr("b", (100,)), _arr("c", (50,))])
        panels = build_panels(s)
        overlays = [p for p in panels if p.kind == "overlay"]
        standalones = [p for p in panels if p.kind == "standalone"]
        assert len(overlays) == 1 and overlays[0].series == ["a", "b"]
        assert len(standalones) == 1
        assert standalones[0].series == ["c"]

    def test_twos_each_get_own_grid_panel(self):
        s = make_state([_arr("m1", (10, 20)), _arr("m2", (5, 5))])
        panels = build_panels(s)
        grids = [p for p in panels if p.kind == "grid"]
        assert len(grids) == 2
        assert {g.array2d_key for g in grids} == {"m1", "m2"}

    def test_mixed_distributions(self):
        s = make_state(
            [
                _arr("a", (100,)),
                _arr("b", (100,)),
                _arr("c", (50,)),
                _arr("m", (10, 20)),
                _arr("v", (5, 10, 20)),
            ]
        )
        panels = build_panels(s)
        kinds_ = kinds(panels)
        assert kinds_.count("overlay") == 1
        assert kinds_.count("standalone") == 1
        assert kinds_.count("grid") == 1
        assert kinds_.count("slice") == 1


# --------------------------------------------------------------------- X/Y


class TestXyMode:
    def test_y_series_are_matching_ones(self):
        s = make_state(
            [_arr("t", (200,)), _arr("a", (200,)), _arr("b", (200,))],
            x_key="t",
        )
        panels = build_panels(s)
        xy = [p for p in panels if p.kind == "xy"]
        assert len(xy) == 1
        assert xy[0].x_key == "t"
        assert set(xy[0].series) == {"a", "b"}

    def test_length_mismatch_excluded_with_reason(self):
        s = make_state(
            [_arr("t", (200,)), _arr("a", (200,)), _arr("c", (150,))],
            x_key="t",
        )
        panels = build_panels(s)
        xy = [p for p in panels if p.kind == "xy"]
        standalone = [p for p in panels if p.kind == "standalone"]
        assert xy[0].series == ["a"]
        assert xy[0].mismatched == [("c", 200, 150)]
        assert len(standalone) == 1
        assert standalone[0].series == ["c"]
        # The standalone panel carries the exclusion reason too.
        assert standalone[0].mismatched == [("t", 200, 150)]

    def test_default_index_mode_when_no_x(self):
        s = make_state(
            [_arr("t", (200,)), _arr("a", (200,)), _arr("b", (200,))],
            x_key=None,
        )
        panels = build_panels(s)
        assert panels[0].kind == "overlay"
        assert panels[0].x_key is None


# --------------------------------------------------------------------- 3-D+


class TestSliceCards:
    def test_high_dim_gets_badged_slice_card(self):
        s = make_state([_arr("v", (5, 10, 20))])
        panels = build_panels(s)
        assert len(panels) == 1
        p = panels[0]
        assert p.kind == "slice"
        assert p.array2d_key == "v"
        assert p.slice_index == 0
        assert "slice [0]" in p.title

    def test_slice_defaults_to_heatmap(self):
        s = make_state([_arr("v", (5, 10, 20))])
        assert build_panels(s)[0].plot_type == "heatmap"


# --------------------------------------------------------------------- types


class TestPlotTypeOverrides:
    def test_grid_override_heatmap_to_contour(self):
        s = make_state([_arr("m", (10, 20))], panel_types={"m": "contour"})
        p = build_panels(s)[0]
        assert p.kind == "grid"
        assert p.plot_type == "contour"

    def test_line_override_applies_to_standalone(self):
        # Standalone mismatch panels only exist in X/Y mode.
        s = make_state(
            [_arr("a", (100,)), _arr("b", (50,))],
            x_key="a",
            panel_types={"b": "scatter"},
        )
        standalone = [p for p in build_panels(s) if p.kind == "standalone" and p.series == ["b"]]
        assert len(standalone) == 1
        assert standalone[0].plot_type == "scatter"

    def test_invalid_override_falls_back_to_default(self):
        s = make_state([_arr("m", (10, 20))], panel_types={"m": "not-a-type"})
        assert build_panels(s)[0].plot_type == "heatmap"


# ---------------------------------------------------------------- non-numeric


class TestNonNumericExclusion:
    """Non-numeric dtypes stay in the checklist but produce no panel."""

    def test_string_1d_gets_no_panel(self):
        s = make_state(
            [_arr("a", (100,)), _arr("labels", (3,), dtype="U1")],
            selected={"a", "labels"},
        )
        panels = build_panels(s)
        assert len(panels) == 1
        assert panels[0].kind == "standalone"
        assert panels[0].series == ["a"]

    def test_string_2d_gets_no_grid_panel(self):
        s = make_state(
            [_arr("m", (10, 20)), _arr("table", (4, 3), dtype="U5")],
            selected={"m", "table"},
        )
        panels = build_panels(s)
        assert [p.kind for p in panels] == ["grid"]
        assert panels[0].array2d_key == "m"

    def test_only_non_numeric_selected_yields_no_panels(self):
        s = make_state([_arr("labels", (3,), dtype="U1")])
        assert build_panels(s) == []

    def test_bool_1d_gets_no_panel(self):
        s = make_state([_arr("flags", (10,), dtype=bool)])
        assert build_panels(s) == []


# --------------------------------------------------------------------- ids


class TestPanelIdStability:
    def test_stable_key_per_panel_kind(self):
        s = make_state([_arr("a", (100,)), _arr("b", (100,))])
        panels = build_panels(s)
        keys = [p._stable_key() for p in panels]
        assert len(keys) == len(set(keys))  # unique across re-render
