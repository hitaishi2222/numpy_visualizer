"""Unit tests for npz_visualizer.npz_store (scan / open / cache / stats)."""

from __future__ import annotations

import os
import time
import zipfile

import numpy as np
import pytest

from npz_visualizer import npz_store
from npz_visualizer.npz_store import (
    NpzCorruptError,
    NpzPathError,
    NpzPickleError,
    compute_stats,
    evict_all,
    get_array,
    get_stats,
    open_npz,
    preview_values,
    scan_dir,
)


@pytest.fixture(autouse=True)
def clean_caches():
    evict_all()
    yield
    evict_all()


def make_npz(dirpath, name: str, **arrays) -> str:
    path = dirpath / name
    np.savez(path, **arrays)
    return str(path)


# --------------------------------------------------------------------------- scan


class TestScanDir:
    def test_lists_npz_files_sorted_with_sizes(self, tmp_path):
        make_npz(tmp_path, "b.npz", x=np.ones(3))
        make_npz(tmp_path, "a.npz", y=np.ones(5))
        make_npz(tmp_path, "c.npz", z=np.ones(7))
        (tmp_path / "notes.txt").write_text("not an npz")

        entries = scan_dir(tmp_path)

        assert [e.name for e in entries] == ["a.npz", "b.npz", "c.npz"]
        assert all(e.size > 0 for e in entries)
        assert all(isinstance(e.mtime, float) and e.mtime > 0 for e in entries)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert scan_dir(tmp_path) == []

    def test_missing_path_raises_named_error(self, tmp_path):
        with pytest.raises(NpzPathError, match="not found"):
            scan_dir(tmp_path / "does-not-exist")

    def test_non_directory_path_raises(self, tmp_path):
        f = tmp_path / "afile.npz"
        make_npz(tmp_path, "afile.npz", x=np.ones(1))
        with pytest.raises(NpzPathError, match="Not a directory"):
            scan_dir(f)


# --------------------------------------------------------------------------- open


class TestOpenNpz:
    def test_valid_multi_key_file_metadata(self, tmp_path):
        path = make_npz(
            tmp_path,
            "mix.npz",
            x=np.arange(100, dtype=np.float64),
            matrix=np.ones((10, 20), dtype=np.float32),
            counts=np.arange(5, dtype=np.int32),
            labels=np.array(["a", "bb", "ccc"], dtype="U3"),
        )
        handle = open_npz(path)

        assert handle.path == path
        assert len(handle.arrays) == 4
        by_name = {m.name: m for m in handle.arrays}

        x = by_name["x"]
        assert (x.shape, x.dtype, x.ndim) == ([100], "float64", 1)
        assert x.n_elements == 100
        assert x.nbytes == 100 * 8
        assert x.stats is None  # not computed until requested
        assert x.preview is not None and len(x.preview) == 10

        m = by_name["matrix"]
        assert (m.shape, m.dtype, m.ndim) == ([10, 20], "float32", 2)
        assert m.n_elements == 200
        assert m.nbytes == 200 * 4
        assert m.preview is None  # 2-D: no preview

        c = by_name["counts"]
        assert (c.shape, c.dtype) == ([5], "int32")
        assert c.stats_na is False  # integers get real stats

        s = by_name["labels"]
        assert s.stats_na is True  # strings -> "n/a" in UI
        assert s.preview == ["a", "bb", "ccc"]

    def test_corrupt_file_raises(self, tmp_path):
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"this is not a zip archive at all")
        with pytest.raises(NpzCorruptError):
            open_npz(bad)

    def test_zip_without_arrays_raises(self, tmp_path):
        empty_zip = tmp_path / "empty.npz"
        with zipfile.ZipFile(empty_zip, "w") as zf:
            zf.writestr("readme.txt", "no arrays here")
        with pytest.raises(NpzCorruptError):
            open_npz(empty_zip)

    def test_object_dtype_refused(self, tmp_path):
        path = make_npz(tmp_path, "obj.npz", thing=np.array([{"a": 1}], dtype=object))
        with pytest.raises(NpzPickleError, match="not supported"):
            open_npz(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(NpzPathError, match="not found"):
            open_npz(tmp_path / "nope.npz")


# --------------------------------------------------------------------------- stats


class TestComputeStats:
    def test_float_stats_exact(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = compute_stats(arr)
        assert (s.min, s.max, s.mean) == (1.0, 5.0, 3.0)
        assert s.std == pytest.approx(np.sqrt(2.0))

    def test_int64_no_overflow(self):
        big = np.array([np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max], dtype=np.int64)
        s = compute_stats(big)
        assert s.max == float(np.iinfo(np.int64).max)
        assert s.min == float(np.iinfo(np.int64).max - 1)

    def test_non_numeric_returns_none(self):
        assert compute_stats(np.array(["x", "y"], dtype="U2")) is None
        assert compute_stats(np.array([True, False])) is None
        assert compute_stats(np.array([1 + 2j, 3 + 4j], dtype=np.complex128)) is None

    def test_empty_array_returns_none(self):
        assert compute_stats(np.array([], dtype=np.float64)) is None


# --------------------------------------------------------------------------- cache


class TestCache:
    def test_get_array_caches_by_identity(self, tmp_path):
        path = make_npz(tmp_path, "c.npz", x=np.arange(10.0))
        a1 = get_array(path, "x")
        a2 = get_array(path, "x")
        assert a1 is a2  # served from cache
        assert len(npz_store._array_cache) == 1

    def test_mtime_change_evicts_stale_entries(self, tmp_path):
        path = make_npz(tmp_path, "m.npz", x=np.arange(10.0))
        first = get_array(path, "x")
        mtime = os.path.getmtime(path)
        # Simulate the file being rewritten (newer mtime).
        os.utime(path, (mtime + 5, mtime + 5))
        second = get_array(path, "x")
        assert second is not first
        keys = list(npz_store._array_cache.keys())
        assert len(keys) == 1 and keys[0][1] > mtime

    def test_evict_all_clears(self, tmp_path):
        path = make_npz(tmp_path, "e.npz", x=np.arange(4.0))
        get_array(path, "x")
        evict_all()
        assert npz_store._array_cache == {} and npz_store._stats_cache == {}

    def test_get_stats_cached(self, tmp_path):
        path = make_npz(tmp_path, "s.npz", x=np.arange(10.0))
        s1 = get_stats(path, "x")
        assert s1 is not None and s1.mean == 4.5
        assert len(npz_store._stats_cache) == 1
        s2 = get_stats(path, "x")
        assert s2 == s1  # second call served from stats cache

    def test_get_array_missing_key_raises(self, tmp_path):
        path = make_npz(tmp_path, "k.npz", x=np.arange(3.0))
        with pytest.raises(NpzCorruptError):
            get_array(path, "nope")


# --------------------------------------------------------------------------- preview


class TestPreview:
    def test_preview_bounded_to_ten(self, tmp_path):
        path = make_npz(tmp_path, "p.npz", x=np.arange(50.0))
        assert preview_values(path, "x") == [float(i) for i in range(10)]

    def test_preview_short_array(self, tmp_path):
        path = make_npz(tmp_path, "p2.npz", x=np.array([1.0, 2.0]))
        assert preview_values(path, "x") == [1.0, 2.0]
