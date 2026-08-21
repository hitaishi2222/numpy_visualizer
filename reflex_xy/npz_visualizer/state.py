"""Dashboard state for the NPZ visualizer (design D2).

Holds only lightweight, JSON-safe metadata. Raw ``ndarray``s live in the
server-side cache in :mod:`npz_visualizer.npz_store`, keyed by
``(path, mtime, key)`` and never round-tripped through Reflex state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import reflex as rx
from pydantic import BaseModel
from reflex.state import computed_var

from npz_visualizer.panels import (
    GRID_2D,
    LINE_1D,
    SLOT_COUNT,
    build_panels,
    state_key,
)
from npz_visualizer.h5_store import (
    ArrayMeta,
    FileEntry,
    NpzError,
    evict_all,
    format_preview,
    format_size,
    open_array_file,
    scan_dir,
)


class ArrayRow(BaseModel):
    """One pre-displayed checklist row (see :attr:`Dashboard.array_rows`).

    Presentational strings (shape, human size, 1-D preview, stats) are all
    computed here so the UI only reads plain fields — ``foreach`` in this
    reflex version hands items as typed Vars, never concrete instances.
    """

    name: str
    ndim: int
    shape: str
    dtype: str
    size: str
    preview: str
    computing: bool
    has_stats: bool
    stats_na: bool
    stats_text: str
    # Presentation helpers (precomputed so the UI stays dumb).
    dim_label: str = ""
    dim_color: str = "indigo"


class Dashboard(rx.State):
    """Single-page dashboard state."""

    # --- directory / file list ------------------------------------------
    dir_path: str = ""
    scan_error: str | None = None
    files: list[FileEntry] = []

    # --- open file ------------------------------------------------------
    open_file: str | None = None
    open_error: str | None = None
    arrays: list[ArrayMeta] = []

    # --- selection ------------------------------------------------------
    selected: set[str] = set()
    x_key: str | None = None
    panel_types: dict[str, str] = {}

    # --- lazy stats (server-computed, shown with a loading indicator) ----
    stats_data: dict[str, dict] = {}
    computing_stats: set[str] = set()

    # ---------------------------------------------------------- bind helpers
    @rx.event
    def set_dir_path(self, path: str) -> None:
        """Set the directory path from the sidebar text input."""
        self.dir_path = path

    # ---------------------------------------------------------------- scan
    @rx.event
    def scan(self) -> None:
        """Scan ``dir_path`` for ``*.npz`` files.

        Reset rule (3.2): rescanning replaces the file list and clears any
        previously opened file, selection, X assignment, and type overrides.
        """
        self.scan_error = None
        self.files = []
        try:
            self.files = scan_dir(self.dir_path)
        except NpzError as exc:
            self.files = []
            self.scan_error = str(exc)
        self._reset_open_state()

    # --------------------------------------------------------------- open
    @rx.event
    def open_file_by_name(self, name: str) -> None:
        """Open one array from the file list.

        Reset rule (3.2): opening a file clears the previous selection, X
        assignment, and type overrides (the new file's arrays may differ).
        """
        self.open_error = None
        full_path = str(Path(self.dir_path) / name)
        try:
            handle = open_array_file(full_path)
        except NpzError as exc:
            self.open_file = name
            self.open_error = str(exc)
            self.arrays = []
            self.selected = set()
            self.x_key = None
            self.panel_types = {}
            return
        # Fresh file: free the previous file's cached arrays.
        evict_all()
        self.open_file = name
        self.open_error = None
        self.arrays = handle.arrays
        self.selected = set()
        self.x_key = None
        self.panel_types = {}

    # --------------------------------------------------------- selection
    @rx.event
    def toggle_selection(self, name: str) -> None:
        # Reassign rather than mutate in place: Reflex only marks a field dirty
        # on assignment (__setattr__), so ``self.selected.add(...)`` would not
        # invalidate dependent computed vars (has_panels / slot_* lists).
        nxt = set(self.selected)
        if name in nxt:
            nxt.discard(name)
        else:
            nxt.add(name)
        self.selected = nxt
        self._coerce_x_key()

    @rx.event
    def select_all(self) -> None:
        """Select every 1-D array (the plottable-by-default set)."""
        self.selected = {a.name for a in self.arrays if a.ndim == 1}
        self._coerce_x_key()

    @rx.event
    def clear_selection(self) -> None:
        self.selected = set()
        self.x_key = None

    @rx.event
    def set_x_key(self, name: str) -> None:
        """Designate *name* as the X array for the 1-D group."""
        arr = next(
            (a for a in self.arrays if a.name == name and a.ndim == 1 and name in self.selected),
            None,
        )
        self.x_key = name if arr is not None else None

    @rx.event
    def set_x_or_clear(self, name: str) -> None:
        """X picker handler: an empty selection returns to index mode."""
        if not name:
            self.x_key = None
        else:
            self.set_x_key(name)

    @rx.event
    def set_panel_type(self, name: str, plot_type: str) -> None:
        """Store a per-array plot-type override, or clear it when unset."""
        if plot_type in LINE_1D or plot_type in GRID_2D:
            self.panel_types[name] = plot_type
        else:
            self.panel_types.pop(name, None)

    # ---------------------------------------------------------- helpers
    def _coerce_x_key(self) -> None:
        """Drop ``x_key`` if it no longer names a selected 1-D array."""
        if self.x_key is not None and self.x_key not in self.selected:
            arr = next((a for a in self.arrays if a.name == self.x_key), None)
            if arr is None or arr.ndim != 1:
                self.x_key = None

    @computed_var
    def array_rows(self) -> list[ArrayRow]:
        """Concrete, pre-displayed rows for the array checklist.

        ``foreach`` hands the UI reactive Vars for items, so the (somewhat
        stringy) presentation — shape string, human size, 1-D preview, and the
        lazy stats — is computed here where the :class:`ArrayMeta` objects and
        the stats cache are still concrete (the UI can never do dynamic dict
        key access on a Var).
        """
        rows = []
        for a in self.arrays:
            stats = self.stats_data.get(a.name)
            if a.ndim == 1:
                dim_label, dim_color = "1-D", "indigo"
            elif a.ndim == 2:
                dim_label, dim_color = "2-D", "cyan"
            else:
                dim_label, dim_color = f"{a.ndim}-D", "orange"
            rows.append(
                ArrayRow(
                    name=a.name,
                    ndim=a.ndim,
                    shape=" × ".join(str(d) for d in a.shape) if a.shape else "scalar",
                    dtype=a.dtype,
                    size=format_size(a.nbytes),
                    preview=format_preview(a.preview) if (a.ndim == 1 and a.preview) else "",
                    computing=a.name in self.computing_stats,
                    has_stats=stats is not None,
                    stats_na=bool(stats and stats.get("na")) if stats else a.stats_na,
                    stats_text=stats.get("text", "") if stats else "",
                    dim_label=dim_label,
                    dim_color=dim_color,
                )
            )
        return rows

    @computed_var(
        auto_deps=False,
        deps=["open_file", "arrays", "selected", "x_key", "panel_types", "dir_path"],
    )
    def has_panels(self) -> bool:
        """Whether any panels exist for the open file (panel-grid placeholder)."""
        return len(build_panels(self)) > 0

    # --- panel slots (figure tier) ------------------------------------
    # The main grid is a fixed bank of ``SLOT_COUNT`` panel slots; each slot
    # is a ``@reflex_xy.figure`` computed var in :mod:`npz_visualizer.charts`
    # (``figure_0`` .. ``figure_{SLOT_COUNT-1}``). The lists below are the
    # reactive, UI-facing views over those slots: presence, title, kind, the
    # currently-resolved plot type, and the ``panel_types`` key a type
    # override is stored under. Each is a plain ``list`` of JSON-safe values
    # (not ``Component``s) so the UI can index them without the broken
    # "computed var returns a Component" path.
    #
    # Dependencies are declared explicitly: the builder body lives in the
    # Reflex-free :func:`build_panels`, so reflex's bytecode dependency scan
    # sees no attribute access on ``self``. Explicit deps keep the vars dirty
    # (and their figures re-published) whenever selection / X / file / type
    # change.
    _SLOT_DEPS = ["open_file", "arrays", "selected", "x_key", "panel_types", "dir_path"]

    @computed_var(auto_deps=False, deps=_SLOT_DEPS)
    def slot_presence_list(self) -> list[bool]:
        """Per-slot presence: True while a panel occupies that slot."""
        n = len(build_panels(self))
        return [i < n for i in range(SLOT_COUNT)]

    @computed_var(auto_deps=False, deps=_SLOT_DEPS)
    def slot_titles_list(self) -> list[str]:
        """Per-slot panel title (empty string for empty slots)."""
        panels = build_panels(self)
        return [panels[i].title if i < len(panels) else "" for i in range(SLOT_COUNT)]

    @computed_var(auto_deps=False, deps=_SLOT_DEPS)
    def slot_kind_list(self) -> list[str]:
        """Per-slot panel kind ("" for empty slots)."""
        panels = build_panels(self)
        return [panels[i].kind if i < len(panels) else "" for i in range(SLOT_COUNT)]

    @computed_var(auto_deps=False, deps=_SLOT_DEPS)
    def slot_current_type_list(self) -> list[str]:
        """Per-slot effective plot type (honours the stored override)."""
        panels = build_panels(self)
        return [panels[i].plot_type if i < len(panels) else "" for i in range(SLOT_COUNT)]

    @computed_var(auto_deps=False, deps=_SLOT_DEPS)
    def slot_state_key_list(self) -> list[str]:
        """Per-slot ``panel_types`` key used to store the type override."""
        panels = build_panels(self)
        return [state_key(panels[i]) if i < len(panels) else "" for i in range(SLOT_COUNT)]

    @rx.event
    def set_slot_type(self, slot: int, plot_type: str) -> None:
        """Resolve the panel at ``slot`` to its ``state_key`` and delegate to
        :meth:`set_panel_type`, so a plot-type override survives panel rebuilds.

        The slot *index* is the event arg (writing to a state-keyed dict with a
        Var key is unsupported), resolved to the key here.
        """
        panels = build_panels(self)
        if not 0 <= slot < len(panels):
            return
        self.set_panel_type(state_key(panels[slot]), plot_type)

    def _reset_open_state(self) -> None:
        """Clear everything tied to the previously opened file (3.2)."""
        evict_all()
        self.open_file = None
        self.open_error = None
        self.arrays = []
        self.selected = set()
        self.x_key = None
        self.panel_types = {}
        self.stats_data = {}
        self.computing_stats = set()

    # ---------------------------------------------------------- stats
    @rx.event
    async def load_stats(self, name: str) -> None:
        """Compute and cache stats for one array (async => loading UI)."""
        if self.open_file is None:
            return
        if name in self.computing_stats:
            return
        self.computing_stats.add(name)
        await asyncio.sleep(0)  # yield so the loading indicator paints
        full_path = str(Path(self.dir_path) / self.open_file)
        arr = next((a for a in self.arrays if a.name == name), None)
        if arr is None:
            self.stats_data[name] = None
            self.computing_stats.discard(name)
            return
        try:
            from npz_visualizer.h5_store import get_stats

            stats = get_stats(full_path, name)
        except NpzError:
            stats = None
        if arr.stats_na:
            self.stats_data[name] = {"na": True}
        elif stats is None:
            self.stats_data[name] = {"na": True}  # empty / non-numeric
        else:
            self.stats_data[name] = {
                "na": False,
                "text": (
                    f"min {stats.min:g} · max {stats.max:g} · "
                    f"mean {stats.mean:g} · std {stats.std:g}"
                ),
            }
        self.computing_stats.discard(name)

    # -------------------------------------------------------- convenience
    def array_by_name(self, name: str) -> ArrayMeta | None:
        return next((a for a in self.arrays if a.name == name), None)

    @computed_var
    def x_labels(self) -> list[str]:
        """Names of selected 1-D arrays, for the X picker UI (reactive)."""
        return [a.name for a in self.arrays if a.ndim == 1 and a.name in self.selected]

    @computed_var
    def show_x_picker(self) -> bool:
        """Whether the X/Y picker should render (>= 2 selected 1-D arrays)."""
        return len(self.x_labels) >= 2

    @computed_var
    def open_panels(self) -> list[dict]:
        """Panel specs for the currently open file, as JSON dicts (reactive)."""
        return [p.to_dict() for p in build_panels(self)]
