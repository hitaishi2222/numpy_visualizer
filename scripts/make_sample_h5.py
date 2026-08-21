"""Generate ``data/sample.h5`` — a small synthetic HDF5 demo archive.

Contents exercise every panel type:
  * ``signals/raw`` + ``signals/filtered``  — nested group, 1-D overlay + X/Y
  * ``frames/checkerboard``                 — 2-D heatmap
  * ``volume/blob``                         — 3-D Plotly isosurface / scatter3d
  * ``meta/labels``                         — string dataset (stats "n/a")

Run:  uv run python scripts/make_sample_h5.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def main(out: str = "data/sample.h5") -> None:
    rng = np.random.default_rng(7)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, 4.0 * np.pi, 400)
    raw = np.sin(t) + 0.25 * rng.standard_normal(t.size)
    kernel = np.ones(9) / 9.0
    filtered = np.convolve(raw, kernel, mode="same")

    n = 32
    zz, yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j]
    blob = (np.exp(-2.5 * (xx**2 + yy**2 + zz**2)) * (1.0 + 0.3 * np.sin(6 * xx))).astype(
        np.float32
    )

    grid = np.indices((24, 24)).sum(axis=0) % 2
    checker = ((grid * 200) + rng.integers(0, 30, size=(24, 24))).astype(np.uint8)

    with h5py.File(path, "w") as f:
        f.create_dataset("signals/raw", data=raw)
        f.create_dataset("signals/filtered", data=filtered)
        f.create_dataset("signals/time", data=t)
        f.create_dataset("frames/checkerboard", data=checker, compression="gzip")
        f.create_dataset("volume/blob", data=blob, compression="gzip")
        ds = f.create_dataset(
            "meta/labels", data=np.array(["alpha", "beta", "gamma"], dtype=h5py.string_dtype())
        )
        ds.attrs["unit"] = "category"
        f.attrs["generator"] = "scripts/make_sample_h5.py"

    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
