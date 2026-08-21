"""NPZ Visualizer — Reflex app entry point (module matched to app_name)."""

from __future__ import annotations

import reflex as rx
import reflex_xy as rxy
import numpy as np

# Import charts first so its module-level code attaches the @rxy.figure slot
# vars (figure_0 .. figure_7) to Dashboard. This also imports reflex_xy / xy
# (isolated to charts.py).
from npz_visualizer import charts  # noqa: F401  (side effect: registers figure slots)
from npz_visualizer import state  # noqa: F401  (side effect: registers Dashboard)
from npz_visualizer.ui import index

# Minimal test state for debugging chart hydration
class TestChartState(rx.State):
    @rxy.data
    def test_data(self):
        return {'x': np.arange(10, dtype=float), 'y': np.random.rand(10)}

def test_chart_page():
    return rx.vstack(
        rx.heading("Minimal Chart Test"),
        rx.text("If you see this text but no chart below, hydration failed."),
        rxy.chart(
            rxy.line(x='x', y='y', name='test'),
            data=TestChartState.test_data,
            height='300px'
        )
    )

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="indigo",
        gray_color="slate",
        radius="large",
    )
)
app.add_page(index)
app.add_page(test_chart_page, route="/test-chart")
