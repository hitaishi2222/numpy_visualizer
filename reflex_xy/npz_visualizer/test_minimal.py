import reflex as rx
import reflex_xy as rxy
import numpy as np

class TestState(rx.State):
    @rxy.data
    def test_data(self):
        return {'x': np.arange(10, dtype=float), 'y': np.random.rand(10)}

def test_page():
    return rx.vstack(
        rx.heading("Minimal Chart Test"),
        rxy.chart(
            rxy.line(x='x', y='y', name='test'),
            data=TestState.test_data,
            height='300px'
        )
    )

app = rx.App()
app.add_page(test_page, route="/test")

if __name__ == "__main__":
    app.compile()