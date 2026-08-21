from typing import DynamicTypedDict

import numpy as np
import reflex as rx
import reflex_xy as rxy


class AllData(DynamicTypedDict):
    pass


class D(rx.State):
    @rxy.data
    def cols(self) -> AllData:
        return {
            "a": np.sin(np.arange(20) * 0.3),
            "b": np.sin(np.arange(35) * 0.3),
        }  # type: ignore


def index() -> rx.Component:
    # y-only, implicit index, two series of different lengths in one chart
    return rxy.line_chart(
        data=D.cols,
        y=["a", "b"],
        height="200px",
    )


app = rx.App()
app.add_page(index)
