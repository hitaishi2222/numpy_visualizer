"""Tests for the unified .npz + HDF5 store (:mod:`npz_visualizer.h5_store`)."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from npz_visualizer.h5_store import (  # noqa: E402
    NpzCorruptError,
    NpzError,
    get_array,
    get_stats,
    open_array_file,
    scan_dir,
)


@pytest.fixture()
def h5_path(tmp_path):
    p = tmp_path / "sample.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("signals/raw", data=np.linspace(0, 1, 50))
        f.create_dataset("signals/time", data=np.arange(50, dtype=np.float32))
        f.create_dataset("frames/grid", data=np.ones((8, 8), dtype=np.uint8))
        f.create_dataset(
            "meta/labels", data=np.array(["a", "b"], dtype=h5py.string_dtype())
        )
        f.create_dataset("volume/cube", data=np.zeros((6, 6, 6), dtype=np.float32))
    return str(p)


def test_scan_lists_h5_and_npz(tmp_path, h5_path):
    (tmp_path / "other.npz").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("nope")
    names = [e.name for e in scan_dir(str(tmp_path))]
    assert names == ["other.npz", "sample.h5"]


def test_open_flattens_groups(h5_path):
    handle = open_array_file(h5_path)
    names = [a.name for a in handle.arrays]
    assert "signals/raw" in names
    assert "frames/grid" in names
    assert "volume/cube" in names


def test_metadata_and_na_flag(h5_path):
    meta = {a.name: a for a in open_array_file(h5_path).arrays}
    raw = meta["signals/raw"]
    assert raw.shape == [50] and raw.ndim == 1
    assert len(raw.preview) == 10
    labels = meta["meta/labels"]
    assert labels.stats_na is True  # string dtype -> stats n/a
    cube = meta["volume/cube"]
    assert cube.ndim == 3 and cube.shape == [6, 6, 6]


def test_get_array_and_stats(h5_path):
    arr = get_array(h5_path, "signals/time")
    assert arr.dtype == np.float32 and arr.shape == (50,)
    stats = get_stats(h5_path, "signals/time")
    assert stats is not None and stats.min == 0.0
    assert get_stats(h5_path, "meta/labels") is None


def test_corrupt_file_refused(tmp_path):
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5 at all")
    with pytest.raises(NpzCorruptError):
        open_array_file(str(bad))


def test_unsupported_extension(tmp_path):
    weird = tmp_path / "x.mat"
    weird.write_bytes(b"")
    with pytest.raises(NpzError):
        open_array_file(str(weird))
