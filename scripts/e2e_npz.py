"""End-to-end exercise of the npz visualizer pipeline.

Runs headless (no browser): generates a sample ``.npz``, drives
``npz_store`` + ``build_panels`` through the full user journey
scan -> open -> select -> overlay -> X/Y -> type switch -> file-switch reset,
and times a large-array open for the performance check.

Run:  .venv/bin/python scripts/e2e_npz.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

# Allow running as a plain script (``python scripts/e2e_npz.py``) without the
# repo root on sys.path / an installed package. The Reflex app package lives
# in ``reflex_xy/``.
_ROOT = str(Path(__file__).resolve().parents[1])
for _p in (_ROOT, str(Path(_ROOT) / "reflex_xy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from npz_visualizer.npz_store import open_npz, evict_all
from npz_visualizer.panels import build_panels

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
results: list[tuple[str, bool]] = []


def check(label: str, cond: bool) -> None:
    results.append((label, cond))
    print(f"  [{PASS if cond else FAIL}] {label}")


def state(handle, open_name: str, dir_path: str, selected, x_key=None, panel_types=None):
    return SimpleNamespace(
        arrays=handle.arrays,
        open_file=open_name,
        dir_path=dir_path,
        selected=set(selected),
        x_key=x_key,
        panel_types=panel_types or {},
    )


def by_kind(panels, kind):
    return [p for p in panels if p.kind == kind]


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    sample = tmp / "sample.npz"
    np.savez(
        sample,
        t=np.arange(200, dtype=float),
        a=np.sin(np.arange(200) * 0.3),
        b=np.cos(np.arange(200) * 0.3),
        c=np.arange(150, dtype=float),          # mismatched length
        matrix=np.random.rand(10, 20),
        vol=np.random.rand(5, 10, 20),
        labels=np.array(["x", "y", "z"], dtype="U1"),
    )

    print("\n== scan ==")
    from npz_visualizer.npz_store import scan_dir

    scanned = [e.name for e in scan_dir(tmp)]
    check("scan lists the sample file", scanned == ["sample.npz"])

    print("\n== open ==")
    handle = open_npz(sample)
    names = {a.name for a in handle.arrays}
    expected = {"t", "a", "b", "c", "matrix", "vol", "labels"}
    check("open enumerates all 7 arrays", names == expected)
    by_name = {a.name: a for a in handle.arrays}
    check("matrix is 2-D", by_name["matrix"].ndim == 2 and by_name["matrix"].shape == [10, 20])
    check("vol is 3-D", by_name["vol"].ndim == 3)
    check("labels is non-numeric", by_name["labels"].stats_na is True)

    print("\n== index-mode overlay (all 1-D selected) ==")
    s = state(handle, "sample.npz", str(tmp), {"t", "a", "b", "c", "matrix", "vol"})
    panels = build_panels(s)
    overlay = by_kind(panels, "overlay")
    standalone = by_kind(panels, "standalone")
    check("equal-length t,a,b overlay", len(overlay) == 1 and set(overlay[0].series) == {"a", "b", "t"})
    check("mismatched c (len 150) gets own panel", len(standalone) == 1 and standalone[0].series == ["c"])
    check("matrix -> grid panel", [p.array2d_key for p in by_kind(panels, "grid")] == ["matrix"])
    check("vol -> slice card", by_kind(panels, "slice")[0].array2d_key == "vol")
    check("slice defaults to heatmap", by_kind(panels, "slice")[0].plot_type == "heatmap")

    print("\n== X/Y mode (t as X) ==")
    s = state(handle, "sample.npz", str(tmp), {"t", "a", "b", "c"}, x_key="t")
    panels = build_panels(s)
    xy = by_kind(panels, "xy")
    check("one XY panel vs t", len(xy) == 1 and xy[0].x_key == "t")
    check("a and b are Y series", set(xy[0].series) == {"a", "b"})
    check("c excluded with reason", xy[0].mismatched == [("c", 200, 150)])
    check("c also rendered as standalone", [p.series for p in by_kind(panels, "standalone")] == [["c"]])

    print("\n== plot-type override (grid -> contour) ==")
    s = state(handle, "sample.npz", str(tmp), {"matrix"}, panel_types={"matrix": "contour"})
    p = build_panels(s)[0]
    check("matrix panel overridden to contour", p.kind == "grid" and p.plot_type == "contour")

    print("\n== file-switch reset ==")
    evict_all()
    s = state(handle, "sample.npz", str(tmp), {"t", "a"}, x_key="t", panel_types={"matrix": "contour"})
    before = build_panels(s)
    evict_all()  # simulate opening a different file
    s2 = state(handle, "sample.npz", str(tmp), set(), x_key=None, panel_types={})
    after = build_panels(s2)
    check("empty selection + reset -> no panels", after == [])

    print("\n== performance: large arrays ==")
    from npz_visualizer.npz_store import compute_stats, get_array

    big = tmp / "big.npz"
    np.savez(big, huge=np.arange(600_000, dtype=float), big2d=np.random.rand(1000, 1000))
    t0 = time.perf_counter()
    open_npz(big)
    open_secs = time.perf_counter() - t0
    check("open of 600k + 1e6-matrix < 5s", open_secs < 5.0)

    t0 = time.perf_counter()
    arr = get_array(big, "huge")
    get_secs = time.perf_counter() - t0
    check("materialize 600k array < 3s", get_secs < 3.0 and arr.shape == (600_000,))

    t0 = time.perf_counter()
    st = compute_stats(get_array(big, "big2d"))
    big2d_secs = time.perf_counter() - t0
    check("stats on 1e6-matrix < 5s", big2d_secs < 5.0 and st is not None)

    print("\n== summary ==")
    passed = sum(1 for _, c in results if c)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        failed = [l for l, c in results if not c]
        print("  FAILED:", failed)
        raise SystemExit(1)
    print("  All E2E checks passed.")


if __name__ == "__main__":
    main()
