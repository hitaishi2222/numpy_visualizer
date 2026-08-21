"""NPZ Visualizer package.

The Reflex app lives in :mod:`npz_visualizer.npz_visualizer` (the module name
matched to ``app_name`` in ``rxconfig.py``). Importing the package there also
registers the :class:`Dashboard` state and attaches the ``@rxy.data`` chart
method via :mod:`npz_visualizer.charts`.
"""

from __future__ import annotations

from npz_visualizer.npz_visualizer import app  # noqa: F401
