"""Visor en vivo: gráfico multicanal rodante para la adquisición en tiempo real.

Mantiene un buffer circular ``(n_canales, ventana)`` y actualiza curvas
persistentes (sin recrearlas) en cada refresco, para que sea fluido a ~30 fps.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .signal_view import _CURVE_COLORS


class LiveSignalView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._buffer: np.ndarray | None = None
        self._fs = 128.0
        self._win = 640
        self._channels: list[str] = []
        self._curves: list[pg.PlotDataItem] = []
        self._spacing = 4.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Tiempo", units="s")
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.setClipToView(True)
        layout.addWidget(self.plot)

    def configure(self, channel_names: list[str], fs: float, window_seconds: float = 5.0) -> None:
        self._fs = float(fs)
        self._channels = list(channel_names)
        self._win = max(64, int(window_seconds * self._fs))
        n = len(channel_names)
        self._buffer = np.zeros((n, self._win), dtype=np.float64)

        self.plot.clear()
        self._curves = []
        ticks = []
        x = np.linspace(-window_seconds, 0.0, self._win)
        for i in range(n):
            offset = (n - 1 - i) * self._spacing
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            curve = self.plot.plot(x, np.full(self._win, offset), pen=pg.mkPen(color, width=1))
            self._curves.append(curve)
            ticks.append((offset, channel_names[i]))
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setLabel("left", "Canal")
        self.plot.setXRange(-window_seconds, 0.0, padding=0.01)

    def append(self, chunk: np.ndarray) -> None:
        if self._buffer is None or chunk is None or chunk.size == 0:
            return
        k = chunk.shape[1]
        if k >= self._win:
            self._buffer = chunk[:, -self._win:].astype(np.float64)
        else:
            self._buffer = np.roll(self._buffer, -k, axis=1)
            self._buffer[:, -k:] = chunk
        self._redraw()

    def _redraw(self) -> None:
        buf = self._buffer
        n = buf.shape[0]
        x = np.linspace(-self._win / self._fs, 0.0, self._win)
        mean = buf.mean(axis=1, keepdims=True)
        std = buf.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        disp = (buf - mean) / std
        for i in range(n):
            offset = (n - 1 - i) * self._spacing
            self._curves[i].setData(x, disp[i] + offset)

    def clear(self) -> None:
        if self._buffer is not None:
            self._buffer[:] = 0.0
            self._redraw()
