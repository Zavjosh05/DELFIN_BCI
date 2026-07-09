"""Visualización de las características extraídas de un segmento/selección.

Muestra una tabla **canales × características** (potencias por banda + temporales)
con las celdas coloreadas según su valor relativo dentro de cada columna (mapa de
calor), para ver de un vistazo qué canales y bandas destacan.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_STOPS = [(0.0, (27, 40, 56)), (0.5, (38, 166, 154)), (1.0, (255, 213, 79))]


def _heat_color(norm: float) -> QColor:
    """Color de mapa de calor (oscuro→teal→amarillo) para ``norm`` en [0,1]."""
    norm = max(0.0, min(1.0, float(norm)))
    for k in range(len(_STOPS) - 1):
        x0, c0 = _STOPS[k]
        x1, c1 = _STOPS[k + 1]
        if norm <= x1:
            f = (norm - x0) / (x1 - x0) if x1 > x0 else 0.0
            return QColor(*[int(a + (b - a) * f) for a, b in zip(c0, c1)])
    return QColor(*_STOPS[-1][1])


def build_feature_table(channel_names: list[str], band_powers: dict, time_features: dict) -> QTableWidget:
    """Construye la tabla de características coloreada (canales × características)."""
    cols = list(band_powers) + list(time_features)
    n_ch = len(channel_names)
    matrix = np.zeros((n_ch, len(cols)))
    for j, b in enumerate(band_powers):
        matrix[:, j] = np.asarray(band_powers[b])[:n_ch]
    for j, f in enumerate(time_features):
        matrix[:, len(band_powers) + j] = np.asarray(time_features[f])[:n_ch]

    table = QTableWidget(n_ch, len(cols))
    table.setHorizontalHeaderLabels(cols)
    table.setVerticalHeaderLabels(channel_names)
    for j in range(len(cols)):
        col = matrix[:, j]
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = (hi - lo) or 1.0
        for i in range(n_ch):
            v = float(col[i])
            color = _heat_color((v - lo) / rng)
            item = QTableWidgetItem(f"{v:.3g}")
            item.setBackground(QBrush(color))
            lum = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            item.setForeground(QBrush(QColor("#000000" if lum > 140 else "#ffffff")))
            table.setItem(i, j, item)
    table.resizeColumnsToContents()
    return table


def show_feature_dialog(parent, channel_names, band_powers, time_features) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle("Características extraídas de la selección")
    dlg.resize(860, 540)
    lay = QVBoxLayout(dlg)
    lay.addWidget(QLabel(
        "Valor por canal. El color es el valor **relativo dentro de cada columna** "
        "(oscuro = bajo, amarillo = alto). Potencias de banda en µV²/Hz; las "
        "temporales en sus unidades."))
    lay.addWidget(build_feature_table(channel_names, band_powers, time_features))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.reject)
    buttons.accepted.connect(dlg.accept)
    lay.addWidget(buttons)
    dlg.exec()
