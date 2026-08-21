"""Unified array-file store: ``.npz`` plus HDF5 (``.h5`` / ``.hdf5``).

Wraps :mod:`npz_store` and adds an HDF5 reader with the same API surface so
callers (Reflex state / FastAPI routes) never care about the container
format:

* :func:`scan_dir` lists every supported archive extension.
* :func:`open_array_file` returns lazy metadata for every array. HDF5 groups
  are walked recursively and flattened to slash-separated path keys
  (``"group/subgroup/dataset"``); soft/external links are skipped.
* :func:`get_array` / :func:`get_stats` share the ``(path, mtime, key)``
  cache from :mod:`npz_store`, so eviction semantics are identical.

HDF5 notes: chunked/compressed datasets are handled transparently by h5py;
variable-length / string dtypes have kind ``O`` or ``U``/``S`` and are marked
``stats_na`` exactly like non-numeric ``.npz`` members. There is no pickle
execution risk for HDF5, so no analogue of ``allow_pickle=False`` is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import h5py

from htmx_fastapi.npz_store import (
    PREVIEW_LEN,
    ArrayMeta,
    ArrayStats,
    FileEntry,
    NpzCorruptError,
    NpzError,
    NpzPathError,
    _is_numeric,
    _prune,
    _stats_cache,
    _array_cache,
    compute_stats,
    evict_all,
    format_preview,
    format_size,
    get_stats as _npz_get_stats,
    get_array as _npz_get_array,
    open_npz,
)

__all__ = [
    "SUPPORTED_EXTS",
    "ArrayMeta",
    "ArrayStats",
    "FileEntry",
    "NpzError",
    "NpzPathError",
    "NpzCorruptError",
    "compute_stats",
    "evict_all",
    "format_preview",
    "format_size",
    "scan_dir",
    "open_array_file",
    "get_array",
    "get_stats",
]

#: Archive extensions the visualizer understands.
SUPPORTED_EXTS = (".npz", ".h5", ".hdf5")


# --------------------------------------------------------------------------- scanning
def scan_dir(path: str | Path) -> list[FileEntry]:
    """List supported archive files directly inside *path*, sorted by name."""
    p = Path(path).expanduser()
    if not p.exists():
        raise NpzPathError(f"Directory not found: {path}")
    if not p.is_dir():
        raise NpzPathError(f"Not a directory: {path}")
    entries: list[FileEntry] = []
    try:
        files = [f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(files, key=lambda f: f.name):
            st = f.stat()
            entries.append(FileEntry(name=f.name, size=st.st_size, mtime=st.st_mtime))
    except OSError as exc:
        raise NpzPathError(f"Cannot read directory {path}: {exc}") from exc
    return entries


# --------------------------------------------------------------------------- hdf5
def _iter_h5_datasets(h5file) -> list[tuple[str, object]]:
    """All datasets in the file as ``(slash_path, dataset)`` pairs.

    ``visititems`` walks nested groups and naturally skips soft/external
    links (they are neither Group nor Dataset instances when fetched).
    """
    found: list[tuple[str, object]] = []

    def _cb(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append((name, obj))

    h5file.visititems(_cb)
    return found


def _preview_values_h5(ds, n: int = PREVIEW_LEN) -> list:
    """First *n* values of a 1-D HDF5 dataset (bounded read)."""
    count = min(n, int(ds.shape[0]))
    values = ds[:count]
    out = []
    for v in values.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(v)
    return out


def open_h5(path: str | Path):
    """Open an HDF5 file and return lazy metadata for every dataset."""
    import h5py

    p = Path(path).expanduser()
    if not p.exists():
        raise NpzPathError(f"File not found: {path}")
    try:
        hf = h5py.File(p, "r")
    except OSError as exc:
        raise NpzCorruptError(f"Not a valid HDF5 file: {p.name} ({exc})") from exc

    with hf:
        entries = _iter_h5_datasets(hf)
        if not entries:
            raise NpzCorruptError(f"No datasets found in file: {p.name}")
        arrays: list[ArrayMeta] = []
        for key, ds in entries:
            shape = tuple(int(d) for d in ds.shape)
            dtype = ds.dtype
            n_elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            itemsize = getattr(dtype, "itemsize", 0) or 0
            meta = ArrayMeta(
                name=key,
                shape=list(shape),
                dtype=str(dtype),
                ndim=len(shape),
                n_elements=n_elements,
                nbytes=int(n_elements * itemsize),
                stats_na=not _is_numeric(dtype),
            )
            if meta.ndim == 1 and meta.n_elements:
                try:
                    meta.preview = _preview_values_h5(ds)
                except NpzError:
                    meta.preview = None
            arrays.append(meta)

    st = p.stat()
    evict_all()  # switching files frees the previous file's cached arrays
    from htmx_fastapi.npz_store import NpzFileHandle

    return NpzFileHandle(path=str(p), mtime=st.st_mtime, arrays=arrays)


# --------------------------------------------------------------------------- dispatch
def open_array_file(path: str | Path):
    """Open ``.npz`` / ``.h5`` / ``.hdf5`` and return lazy array metadata."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return open_npz(path)
    if suffix in (".h5", ".hdf5"):
        return open_h5(path)
    raise NpzError(f"Unsupported file type: {path}")


def get_array(path: str | Path, key: str) -> np.ndarray:
    """Materialize one array into the shared (path, mtime, key) cache."""
    if Path(path).suffix.lower() == ".npz":
        return _npz_get_array(path, key)
    import h5py

    p = Path(path)
    _prune(p)
    mtime = p.stat().st_mtime
    k = (str(p), mtime, key)
    arr = _array_cache.get(k)
    if arr is None:
        try:
            with h5py.File(p, "r") as hf:
                if key not in hf:
                    raise KeyError(key)
                arr = np.asarray(hf[key][...])
        except (OSError, KeyError) as exc:
            raise NpzCorruptError(f"Cannot read dataset {key!r}: {exc}") from exc
        _array_cache[k] = arr
    return arr


def get_stats(path: str | Path, key: str) -> ArrayStats | None:
    """Stats for one array, computed lazily and cached like the array."""
    if Path(path).suffix.lower() == ".npz":
        return _npz_get_stats(path, key)
    return compute_stats(get_array(path, key))
