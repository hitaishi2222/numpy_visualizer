"""Layout for the NPZ visualizer: header bar + sidebar + panel grid (design D9).

The top bar carries the brand and the currently open file. The sidebar holds
the directory scanner, file list, array checklist, and X picker. The main area
renders one panel card per fixed slot of the figure-tier panel grid; each
non-empty slot shows its title, a kind badge, a plot-type selector, and a
``reflex_xy.chart`` bound to the corresponding ``Dashboard.figure_i`` var.
All chart construction stays in :mod:`npz_visualizer.charts`.
"""

from __future__ import annotations

import reflex as rx

from npz_visualizer import charts, state as state_module
from npz_visualizer.charts import SLOT_COUNT, figure_component
from npz_visualizer.panels import GRID_2D, LINE_1D

Dashboard = state_module.Dashboard

_SIDEBAR_WIDTH = "340px"
_TOPBAR_HEIGHT = "64px"

_SURFACE = "var(--gray-a2)"
_BORDER = "1px solid var(--gray-a5)"


# ------------------------------------------------------------------ helpers
def _section_label(text: str, icon: str) -> rx.Component:
    """Small uppercase section heading with a leading icon."""
    return rx.hstack(
        rx.icon(icon, size=13, color="var(--accent-9)"),
        rx.text(
            text,
            size="1",
            weight="bold",
            letter_spacing="0.1em",
            color="var(--gray-11)",
        ),
        spacing="2",
        align="center",
    )


# -------------------------------------------------------------------- header
def _brand_mark() -> rx.Component:
    return rx.box(
        rx.icon("chart-spline", size=16, color="white"),
        background="linear-gradient(135deg, #6366f1, #8b5cf6)",
        border_radius="10px",
        padding="8px",
        display="flex",
        align_items="center",
        justify_content="center",
        box_shadow="0 2px 8px var(--indigo-a5)",
    )


def top_bar() -> rx.Component:
    """App header: brand on the left, open-file context on the right."""
    return rx.hstack(
        _brand_mark(),
        rx.heading("NPZ Visualizer", size="5", weight="bold", trim="both"),
        rx.badge("beta", color_scheme="gray", variant="surface", size="1", radius="full"),
        rx.spacer(),
        rx.cond(
            Dashboard.open_file != None,  # noqa: E711
            rx.hstack(
                rx.icon("file-archive", size=14, color="var(--accent-9)"),
                rx.badge(
                    Dashboard.open_file,
                    color_scheme="indigo",
                    variant="soft",
                    size="2",
                    radius="full",
                ),
                rx.cond(
                    Dashboard.open_error != None,  # noqa: E711
                    rx.badge(
                        "load error",
                        color_scheme="red",
                        variant="soft",
                        size="2",
                        radius="full",
                    ),
                ),
                spacing="2",
                align="center",
                padding="4px 8px",
                border_radius="10px",
                background=_SURFACE,
                border=_BORDER,
            ),
            rx.hstack(
                rx.icon("circle-dashed", size=14, color="var(--gray-9)"),
                rx.badge(
                    "no file open",
                    color_scheme="gray",
                    variant="soft",
                    size="2",
                    radius="full",
                ),
                spacing="2",
                align="center",
            ),
        ),
        width="100%",
        height=_TOPBAR_HEIGHT,
        align="center",
        padding="0 20px",
        border_bottom=_BORDER,
        background="var(--gray-a3)",
        backdrop_filter="blur(8px)",
        spacing="3",
    )


# ------------------------------------------------------------------ sidebar
def _file_row(entry) -> rx.Component:
    """One file list row; the active file is highlighted."""
    name = entry["name"]
    is_active = Dashboard.open_file == name
    return rx.list_item(
        rx.button(
            rx.icon("file-archive", size=14),
            rx.text(name, overflow="hidden", text_ellipsis="ellipsis"),
            variant=rx.cond(is_active, "solid", "ghost"),
            color_scheme=rx.cond(is_active, "indigo", "gray"),
            width="100%",
            justify_content="flex-start",
            size="1",
            style={"min_width": 0},
            _hover={"background": rx.cond(is_active, "transparent", "var(--gray-a3)")},
            on_click=Dashboard.open_file_by_name(name),
            is_disabled=Dashboard.open_error != None,  # noqa: E711
        ),
        padding_y="1px",
    )


def _stats_row(row) -> rx.Component:
    """Lazy stats with a loading spinner, or a compact 'stats' affordance."""
    return rx.cond(
        row.computing,
        rx.hstack(
            rx.spinner(size="1"),
            rx.text("computing…", size="1", color="var(--gray-10)"),
            spacing="2",
            align="center",
        ),
        rx.cond(
            row.has_stats,
            rx.cond(
                row.stats_na,
                rx.text("stats: n/a (non-numeric)", size="1", color="var(--gray-10)"),
                rx.text(
                    row.stats_text,
                    size="1",
                    font_family="monospace",
                    color="var(--gray-10)",
                ),
            ),
            rx.button(
                "Compute stats",
                size="1",
                variant="ghost",
                color_scheme="indigo",
                on_click=Dashboard.load_stats(row.name),
            ),
        ),
    )


def _array_row(row) -> rx.Component:
    """One checklist row: checkbox + name + dim badge, meta line, preview, stats."""
    return rx.list_item(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.checkbox(
                        checked=Dashboard.selected.contains(row.name),
                        on_change=Dashboard.toggle_selection(row.name),
                        color_scheme="indigo",
                    ),
                    rx.text(
                        row.name,
                        size="2",
                        weight="medium",
                        font_family="monospace",
                        overflow="hidden",
                        text_ellipsis="ellipsis",
                    ),
                    rx.badge(
                        row.dim_label,
                        color_scheme=row.dim_color,
                        variant="soft",
                        size="1",
                        radius="full",
                    ),
                    rx.spacer(),
                    rx.text(row.size, size="1", color="var(--gray-10)"),
                    spacing="2",
                    align="center",
                    width="100%",
                ),
                rx.hstack(
                    rx.text(
                        f"{row.shape}",
                        size="1",
                        color="var(--gray-10)",
                        font_family="monospace",
                    ),
                    rx.text("·", size="1", color="var(--gray-7)"),
                    rx.text(
                        row.dtype,
                        size="1",
                        color="var(--gray-10)",
                        font_family="monospace",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.cond(
                    row.preview != "",  # noqa: E711
                    rx.box(
                        rx.text(
                            row.preview,
                            size="1",
                            font_family="monospace",
                            color="var(--gray-10)",
                            wrap="break",
                        ),
                        padding="4px 8px",
                        border_radius="6px",
                        background="var(--gray-a3)",
                        width="100%",
                    ),
                ),
                _stats_row(row),
                direction="column",
                align="stretch",
                spacing="1",
            ),
            padding="10px 12px",
            width="100%",
            variant="surface",
        ),
        padding_y="3px",
        width="100%",
    )


def sidebar_body() -> rx.Component:
    return rx.vstack(
        # --- directory scan ---
        rx.vstack(
            _section_label("DATA SOURCE", "folder-search"),
            rx.hstack(
                rx.input(
                    placeholder="Path to folder of .npz files",
                    value=Dashboard.dir_path,
                    on_change=Dashboard.set_dir_path,
                    size="2",
                    flex="1",
                ),
                rx.button(
                    rx.icon("folder-search", size=15),
                    "Scan",
                    size="2",
                    color_scheme="indigo",
                    on_click=Dashboard.scan(),
                ),
                spacing="2",
                width="100%",
                align="center",
            ),
            rx.cond(
                Dashboard.scan_error != None,  # noqa: E711
                rx.callout(
                    rx.text(Dashboard.scan_error, size="1"),
                    icon="triangle-alert",
                    color_scheme="red",
                    size="1",
                ),
            ),
            spacing="2",
            align="stretch",
            width="100%",
        ),
        # --- file list ---
        rx.vstack(
            _section_label("FILES", "archive"),
            rx.cond(
                Dashboard.files.length() > 0,
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(Dashboard.files, _file_row),
                        spacing="0",
                        width="100%",
                    ),
                    height="150px",
                    width="100%",
                    type_hint="always",
                ),
                rx.text("No .npz files found.", size="1", color="var(--gray-10)"),
            ),
            spacing="2",
            align="stretch",
            width="100%",
        ),
        rx.divider(size="4", color_scheme="gray"),
        # --- open file / array checklist ---
        rx.cond(
            Dashboard.open_error != None,  # noqa: E711
            rx.callout(
                rx.text(Dashboard.open_error, size="1"),
                icon="triangle-alert",
                color_scheme="red",
                size="1",
            ),
            rx.cond(
                Dashboard.arrays.length() > 0,
                rx.vstack(
                    rx.hstack(
                        _section_label("ARRAYS", "database"),
                        rx.badge(
                            Dashboard.arrays.length(),
                            color_scheme="gray",
                            variant="soft",
                            size="1",
                            radius="full",
                        ),
                        rx.spacer(),
                        rx.button(
                            "All",
                            size="1",
                            variant="soft",
                            color_scheme="indigo",
                            on_click=Dashboard.select_all(),
                        ),
                        rx.button(
                            "Clear",
                            size="1",
                            variant="soft",
                            color_scheme="gray",
                            on_click=Dashboard.clear_selection(),
                        ),
                        spacing="2",
                        align="center",
                        width="100%",
                    ),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(Dashboard.array_rows, _array_row),
                            spacing="1",
                            width="100%",
                        ),
                        height="calc(100vh - 430px)",
                        min_height="240px",
                        width="100%",
                        type_hint="always",
                    ),
                    # --- X picker ---
                    rx.cond(
                        Dashboard.show_x_picker,
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("move-horizontal", size=14),
                                    rx.text(
                                        "X axis (X/Y mode)",
                                        size="1",
                                        weight="bold",
                                    ),
                                    spacing="2",
                                    align="center",
                                ),
                                rx.select(
                                    Dashboard.x_labels,
                                    placeholder="x = index",
                                    value=Dashboard.x_key,
                                    on_change=Dashboard.set_x_or_clear,
                                    size="1",
                                    width="100%",
                                ),
                                spacing="1",
                                align="stretch",
                            ),
                            padding="10px 12px",
                            width="100%",
                            variant="surface",
                            style={"border_left": "2px solid var(--accent-9)"},
                        ),
                    ),
                    direction="column",
                    spacing="2",
                    align="stretch",
                    width="100%",
                ),
                rx.callout(
                    rx.text("Open a file to see its arrays.", size="1"),
                    icon="info",
                    size="1",
                    color_scheme="gray",
                ),
            ),
        ),
        direction="column",
        spacing="4",
        align="stretch",
        width="100%",
    )


def sidebar() -> rx.Component:
    """Directory scanner, file list, array checklist, and X picker."""
    return rx.box(
        rx.scroll_area(
            rx.box(sidebar_body(), padding="16px 14px", width="100%"),
            height=f"calc(100vh - {_TOPBAR_HEIGHT})",
            width="100%",
        ),
        width=_SIDEBAR_WIDTH,
        flex_shrink="0",
        border_right=_BORDER,
        background="var(--gray-a1)",
    )


# -------------------------------------------------------------------- main area
_KIND_BADGES = {
    "overlay": ("overlay", "indigo"),
    "xy": ("X/Y", "violet"),
    "standalone": ("series", "blue"),
    "grid": ("grid", "cyan"),
    "slice": ("slice", "orange"),
}


def _kind_badge(kind) -> rx.Component:
    """Badge for the reactive slot kind (empty string -> render nothing)."""
    parts = []
    for k, (label, color) in _KIND_BADGES.items():
        parts.append(
            rx.cond(
                kind == k,
                rx.badge(label, color_scheme=color, variant="soft", size="1", radius="full"),
                None,
            )
        )
    return rx.hstack(*parts, spacing="1")


def _plot_selector(slot: int) -> rx.Component:
    """Plot-type selector for one panel slot.

    The option set depends on the panel *family*: 1-D panels (overlay / xy /
    standalone) get the line families, 2-D / 3-D+ panels (grid / slice) get the
    grid families. ``kind`` is reactive, so the correct selector is chosen at
    render time. The selected value is written back through
    :meth:`Dashboard.set_slot_type` (the slot index is a fixed arg; the DOM
    value is appended by reflex as ``plot_type``).
    """
    current = Dashboard.slot_current_type_list[slot]
    kind = Dashboard.slot_kind_list[slot]
    is_grid = (kind == "grid") | (kind == "slice")
    line_sel = rx.select(
        list(LINE_1D),
        placeholder="type",
        value=current,
        on_change=Dashboard.set_slot_type(slot),
        size="1",
        width="130px",
    )
    grid_sel = rx.select(
        list(GRID_2D),
        placeholder="type",
        value=current,
        on_change=Dashboard.set_slot_type(slot),
        size="1",
        width="130px",
    )
    # Only one family's options are valid for this panel; show the matching
    # selector and hide the other (it renders as nothing when the cond is false).
    return rx.cond(is_grid, grid_sel, line_sel)


def _slot_card(slot: int) -> rx.Component:
    """One panel card: title + kind badge + type selector, then the chart."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    Dashboard.slot_titles_list[slot],
                    size="2",
                    weight="bold",
                    trim="both",
                ),
                _kind_badge(Dashboard.slot_kind_list[slot]),
                rx.spacer(),
                _plot_selector(slot),
                spacing="2",
                align="center",
                width="100%",
                padding_bottom="8px",
                border_bottom=_BORDER,
            ),
            figure_component(slot),
            direction="column",
            align="stretch",
            spacing="3",
        ),
        width="100%",
        min_width="0",
        variant="surface",
    )


def main_area() -> rx.Component:
    placeholder = rx.card(
        rx.vstack(
            rx.box(
                rx.icon("chart-line", size=36, color="var(--accent-9)"),
                padding="18px",
                border_radius="16px",
                background="var(--accent-a3)",
                display="flex",
            ),
            rx.vstack(
                rx.text("No panels yet", size="4", weight="medium", color="var(--gray-11)"),
                rx.text(
                    "Open a .npz file and select arrays to plot.",
                    size="2",
                    color="var(--gray-10)",
                ),
                spacing="1",
                align="center",
            ),
            direction="column",
            spacing="4",
            align="center",
        ),
        height="70vh",
        display="flex",
        justify_content="center",
        align_items="center",
        width="100%",
        style={
            "border": "1px dashed var(--gray-a7)",
            "border_radius": "16px",
            "background": "var(--gray-a1)",
        },
    )
    # Fixed bank of panel slots; each present slot contributes a card, empty
    # slots collapse to nothing. Indexing the slot lists by a Python int yields
    # a typed Var, which the reactive UI reads directly.
    cards = [
        rx.cond(Dashboard.slot_presence_list[slot], _slot_card(slot), None)
        for slot in range(SLOT_COUNT)
    ]
    grid = rx.grid(
        *cards,
        gap="16px",
        width="100%",
        style={"@media (min-width: 1100px)": {"grid_template_columns": "repeat(2, 1fr)"}},
    )
    return rx.cond(Dashboard.has_panels, grid, placeholder)


def index() -> rx.Component:
    """Header bar over a sidebar | panel-grid layout."""
    return rx.vstack(
        top_bar(),
        rx.hstack(
            sidebar(),
            rx.scroll_area(
                rx.box(main_area(), padding="16px", width="100%"),
                height=f"calc(100vh - {_TOPBAR_HEIGHT})",
                width="100%",
            ),
            spacing="0",
            width="100%",
            align="start",
        ),
        spacing="0",
        width="100%",
        height="100vh",
        overflow="hidden",
    )
