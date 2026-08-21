"""Server-side .npz loading, caching, and statistics.

Design D7: scan stays fast (no per-file open at scan time); opening parses
only the ``.npy`` headers inside the zip so metadata is available without
materializing any array; raw arrays are materialized lazily into a module
cache keyed by ``(path, mtime, key)`` and evicted on file switch / mtime
change. All loads use ``allow_pickle=False`` — object/pickled payloads are
refused up front (their dtype is visible in the header) instead of executed.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel

PREVIEW_LEN = 10


def format_size(nbytes: int) -> str:
    """Human-readable byte size (e.g. ``800B``, ``1.5MB``)."""
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(nbytes)
    for unit in units:
        if val < step or unit == units[-1]:
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)}B"
        val /= step
    return f"{nbytes}B"


def format_preview(values: Sequence[Any] | None) -> str:
    """Compact single-line preview of up to ``PREVIEW_LEN`` values."""
    if not values:
        return ""
    parts = []
    for v in list(values)[:PREVIEW_LEN]:
        if isinstance(v, bytes):
            parts.append(v.decode("utf-8", errors="replace"))
        elif isinstance(v, float):
            parts.append(f"{v:.4g}")
        else:
            parts.append(str(v))
    return ", ".join(parts)


# --------------------------------------------------------------------------- errors

class NpzError(Exception):
    """Base class for .npz handling errors (message is user-facing)."""


class NpzPathError(NpzError):
    """The scanned path does not exist or is not a directory."""


class NpzCorruptError(NpzError):
    """The file cannot be parsed as a .npz archive."""


class NpzPickleError(NpzError):
    """The archive contains object/pickled payloads that are refused."""


# --------------------------------------------------------------------------- models

class FileEntry(BaseModel):
    name: str
    size: int  # bytes
    mtime: float  # epoch seconds


class ArrayStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class ArrayMeta(BaseModel):
    """Lightweight, JSON-safe metadata for one named array of an open file."""

    name: str
    shape: list[int]
    dtype: str
    ndim: int
    n_elements: int
    nbytes: int
    stats: ArrayStats | None = None  # None = not computed yet
    stats_na: bool = False  # True for non-numeric dtypes -> UI shows "n/a"
    preview: list[Any] | None = None  # first <=10 values for 1-D arrays


class NpzFileHandle(BaseModel):
    """Result of opening a file: path plus lazy metadata for every array."""

    path: str
    mtime: float
    arrays: list[ArrayMeta]


# --------------------------------------------------------------------------- scanning

def scan_dir(path: str | Path) -> list[FileEntry]:
    """List ``*.npz`` files directly inside *path*, sorted by name.

    Raises :class:`NpzPathError` for missing/non-directory paths with a
    message that names the problem. An existing directory without matches
    returns an empty list (the UI renders its own empty state).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise NpzPathError(f"Directory not found: {path}")
    if not p.is_dir():
        raise NpzPathError(f"Not a directory: {path}")
    entries: list[FileEntry] = []
    try:
        for f in sorted(p.glob("*.npz"), key=lambda f: f.name):
            st = f.stat()
            entries.append(FileEntry(name=f.name, size=st.st_size, mtime=st.st_mtime))
    except OSError as exc:  # unreadable directory, permission problems, ...
        raise NpzPathError(f"Cannot read directory {path}: {exc}") from exc
    return entries


# --------------------------------------------------------------------------- opening

def _read_npy_header(f):
    """Read one .npy member header; return (shape, fortran_order, dtype).

    Uses numpy's stable low-level header readers (version-dispatched) so we
    never touch the payload bytes.
    """
    from numpy.lib import format as npy_format  # stable internal header parser

    major, minor = npy_format.read_magic(f)
    if (major, minor) == (1, 0):
        return npy_format.read_array_header_1_0(f)
    return npy_format.read_array_header_2_0(f)


def _iter_headers(zf: zipfile.ZipFile):
    """Yield (key, shape, dtype, fortran_order) per .npy member, header-only.

    Reads only the zip central directory and each member's .npy header —
    never the payload bytes — so this stays cheap for huge archives.
    """
    for name in sorted(zf.namelist()):
        if not name.endswith(".npy"):
            continue
        with zf.open(name) as f:
            shape, fortran_order, dtype = _read_npy_header(f)
        yield name[: -len(".npy")], tuple(shape), dtype, bool(fortran_order)


def _is_numeric(dtype: np.dtype) -> bool:
    """Dtypes for which min/max/mean/std make sense (design D7).

    Integer and floating point get real stats; complex/bool/object/string
    show "n/a" in the UI.
    """
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def open_npz(path: str | Path) -> NpzFileHandle:
    """Open *path* and return lazy metadata for every named array.

    Raises :class:`NpzCorruptError` for unreadable/corrupt archives and
    :class:`NpzPickleError` when any entry has object dtype (pickled
    payload) — the whole file is refused, per spec.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise NpzPathError(f"File not found: {path}")
    try:
        zf = zipfile.ZipFile(p)
    except (zipfile.BadZipFile, OSError) as exc:
        raise NpzCorruptError(f"Not a valid .npz archive: {p.name}") from exc

    with zf:
        if not any(n.endswith(".npy") for n in zf.namelist()):
            raise NpzCorruptError(f"No arrays found in archive: {p.name}")
        try:
            headers = list(_iter_headers(zf))
        except (ValueError, EOFError, OSError) as exc:
            raise NpzCorruptError(f"Corrupt .npz header in {p.name}: {exc}") from exc

    pickled = [key for key, _, dtype, _ in headers if dtype == np.dtype("O")]
    if pickled:
        raise NpzPickleError(
            "Pickled/object arrays are not supported "
            f"(found in: {', '.join(pickled)})"
        )

    st = p.stat()
    arrays: list[ArrayMeta] = []
    for key, shape, dtype, _fortran in headers:
        n_elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = int(n_elements * dtype.itemsize)
        arrays.append(
            ArrayMeta(
                name=key,
                shape=list(shape),
                dtype=str(dtype),
                ndim=len(shape),
                n_elements=n_elements,
                nbytes=nbytes,
                stats_na=not _is_numeric(dtype),
            )
        )

    # Preview (first 10 values) for 1-D arrays — bounded read, see below.
    for meta in arrays:
        if meta.ndim == 1:
            try:
                meta.preview = preview_values(p, meta.name)
            except NpzError:
                meta.preview = None

    evict_all()  # switching files frees the previous file's cached arrays
    return NpzFileHandle(path=str(p), mtime=st.st_mtime, arrays=arrays)


def preview_values(path: str | Path, key: str, n: int = PREVIEW_LEN) -> list[Any]:
    """First *n* values of a 1-D array without materializing the whole member.

    .npz members are C-contiguous ``.npy`` files, so the first *n* values sit
    at a fixed byte offset after the header — we seek and read just those
    bytes. Fortran-ordered or exotic layouts fall back to a full (bounded)
    read of the member.
    """
    p = Path(path)
    with zipfile.ZipFile(p) as zf:
        name = key + ".npy"
        if name not in zf.namelist():
            raise NpzCorruptError(f"No such array: {key}")
        total = zf.getinfo(name).file_size
        with zf.open(name) as f:
            shape, fortran_order, dtype = _read_npy_header(f)
            if len(shape) != 1:
                raise NpzError("preview requires a 1-D array")
            count = min(n, int(shape[0]))
            need = count * dtype.itemsize
            offset = f.tell()
            size_on_disk = total - offset
            if fortran_order or need > size_on_disk:
                # Rare layout: read the whole payload once (still bounded by
                # the member size) and take the head.
                raw = f.read()
                arr = np.frombuffer(raw, dtype=dtype)
                values = arr[:count]
            else:
                f.seek(offset)
                raw = f.read(need)
                values = np.frombuffer(raw, dtype=dtype)

    out: list[Any] = []
    for v in values.tolist():
        if isinstance(v, bytes):  # fixed-width strings / byte arrays
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(v)
    return out


# --------------------------------------------------------------------------- cache

# (path, mtime, key) -> ndarray. Bounded to one open file at a time.
_array_cache: dict[tuple[str, float, str], np.ndarray] = {}
_stats_cache: dict[tuple[str, float, str], ArrayStats | None] = {}


def evict_all() -> None:
    """Drop every cached array and stats entry (file switch)."""
    _array_cache.clear()
    _stats_cache.clear()


def _prune(path: Path) -> None:
    """Evict entries that no longer match the file's current (path, mtime)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        evict_all()
        return
    cur_path = str(path)
    for cache in (_array_cache, _stats_cache):
        stale = [k for k in list(cache) if k[0] != cur_path or k[1] != mtime]
        for k in stale:
            del cache[k]


def get_array(path: str | Path, key: str) -> np.ndarray:
    """Materialize one array into the (path, mtime, key)-keyed cache."""
    p = Path(path)
    _prune(p)
    mtime = p.stat().st_mtime
    k = (str(p), mtime, key)
    arr = _array_cache.get(k)
    if arr is None:
        try:
            with np.load(p, allow_pickle=False) as npz:
                if key not in npz.files:
                    raise KeyError(key)
                arr = npz[key]
        except NpzPickleError:
            raise
        except (ValueError, zipfile.BadZipFile, OSError, KeyError) as exc:
            if "allow_pickle" in str(exc):
                raise NpzPickleError(
                    f"Pickled/object array refused: {key}"
                ) from exc
            raise NpzCorruptError(f"Cannot read array {key!r}: {exc}") from exc
        _array_cache[k] = arr
    return arr


def compute_stats(arr: np.ndarray) -> ArrayStats | None:
    """Vectorized min/max/mean/std, or None for non-numeric dtypes.

    Computations run in float64 so 64-bit integer ranges cannot overflow.
    """
    if not _is_numeric(arr.dtype):
        return None
    a = arr.astype(np.float64, copy=False)
    if a.size == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return ArrayStats(
            min=float(a.min()),
            max=float(a.max()),
            mean=float(a.mean()),
            std=float(a.std()),
        )


def get_stats(path: str | Path, key: str) -> ArrayStats | None:
    """Stats for one array, computed lazily and cached like the array."""
    p = Path(path)
    _prune(p)
    mtime = p.stat().st_mtime
    k = (str(p), mtime, key)
    if k in _stats_cache:
        return _stats_cache[k]
    stats = compute_stats(get_array(p, key))
    _stats_cache[k] = stats
    return stats
