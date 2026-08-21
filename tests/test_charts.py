"""Regression tests for chart construction (figure tier).

Covers :func:`npz_visualizer.charts.build_xy_figure`, the runtime, figure-tier
counterpart to the (removed) plan-tier ``build_chart``. Marks carry real numpy
arrays, so structure can vary freely across panels.

The index-mode x-column contract is preserved: xy has no implicit index, so
every index-mode panel (overlay / standalone with ``x_key is None``) must get an
explicit element-index column from ``build_xy_figure`` / ``_index_column`` and
pass it as ``x``. Without it the xy engine fails at probe/runtime with
``ValueError: columns must be 1-D, got shape ()`` and the whole event errors.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npz_visualizer import charts
from npz_visualizer.npz_store import ArrayMeta, open_npz
from npz_visualizer.panels import build_panels


def _arr(name: str, shape, dtype="float64") -> ArrayMeta:
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


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "assets" / "samples"


def _fake_state(arrays, selected, x_key=None):
    """State stand-in with just the fields ``build_panels`` reads."""
    return SimpleNamespace(
        arrays=arrays,
        open_file="demo.npz",
        dir_path=str(SAMPLE_DIR),
        selected=set(selected),
        x_key=x_key,
        panel_types={},
    )


def _is_chart_like(obj) -> bool:
    return callable(getattr(obj, "figure", None)) or callable(
        getattr(obj, "build_payload", None)
    )


@pytest.fixture(scope="module")
def sample():
    """The checked-in demo archive (t/a/b len 200, c len 150, labels U1)."""
    return open_npz(SAMPLE_DIR / "demo.npz")


def _build(sample, selected, x_key=None):
    s = _fake_state(sample.arrays, selected, x_key)
    panels = build_panels(s)
    full = str(SAMPLE_DIR / "demo.npz")
    return s, panels, [charts.build_xy_figure(p, full) for p in panels]


class TestIndexModeColumns:
    """The original crash lived here: index mode must emit a 1-D index x."""

    def test_index_column_is_1d_and_correct_length(self):
        # Explicit _index_column contract: 1-D, float64, length == first series.
        panel = charts.PanelSpec(id="p0", kind="overlay", title="x", series=["a"], x_key=None)
        idx = charts._index_column(panel, str(SAMPLE_DIR / "demo.npz"))
        assert idx.ndim == 1
        assert idx.dtype == np.float64
        assert idx.shape == (200,)
        # never a scalar / 0-D (that was the ``shape ()`` crash)
        assert idx.size == 200

    def test_overlay_index_mode_builds(self, sample):
        s, panels, figures = _build(sample, {"t", "a", "b"})
        assert len(panels) == 1 and panels[0].kind == "overlay"
        assert _is_chart_like(figures[0])

    def test_standalone_index_mode_builds(self, sample):
        s, panels, figures = _build(sample, {"c"})
        assert len(panels) == 1 and panels[0].kind == "standalone"
        assert _is_chart_like(figures[0])


class TestFigureBuild:
    """build_xy_figure must not raise on any panel kind (the original crash)."""

    def test_xy_mode_uses_explicit_x(self, sample):
        s, panels, figures = _build(sample, {"t", "a"}, x_key="t")
        xy = [p for p in panels if p.kind == "xy"][0]
        assert _is_chart_like(figures[0])

    def test_standalone_mismatch_panel_builds(self, sample):
        # c (len 150) is mismatched against t (len 200) -> standalone panel.
        s, panels, figures = _build(sample, {"t", "a", "c"}, x_key="t")
        solo = [p for p in panels if p.kind == "standalone"][0]
        assert solo.series == ["c"]
        assert _is_chart_like(figures[-1])

    def test_grid_panel_builds(self, sample):
        _, panels, figures = _build(sample, {"matrix"})
        assert panels[0].kind == "grid"
        assert _is_chart_like(figures[0])

    def test_slice_panel_builds(self, sample):
        _, panels, figures = _build(sample, {"vol"})
        assert panels[0].kind == "slice"
        assert _is_chart_like(figures[0])


class TestSlotRegistration:
    """The figure slots exist and probe cleanly against default state."""

    def test_figure_slots_registered(self):
        assert charts.SLOT_COUNT == 8
        for slot in range(charts.SLOT_COUNT):
            assert hasattr(charts.Dashboard, f"figure_{slot}")

    def test_default_state_probes_to_empty_handle(self):
        default = charts.Dashboard(_reflex_internal_init=True)
        handle = charts.Dashboard.__dict__["figure_0"].fget(default)
        assert handle.token == ""  # no open file -> no figure -> empty token

    def test_slot_lists_reflect_selection(self, sample):
        # The reactive slot lists are plain Python views over build_panels; here
        # we just assert the underlying panels the lists would enumerate.
        s = _fake_state(sample.arrays, {"t", "a", "b"})
        panels = build_panels(s)
        assert len(panels) == 1
        assert panels[0].kind == "overlay"
        assert panels[0].title.startswith("1-D arrays (index)")
