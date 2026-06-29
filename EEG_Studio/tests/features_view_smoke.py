"""Verifica la tabla de visualización de características (offscreen)."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication

from eeg_studio.core.processing import band_powers, time_features
from eeg_studio.ui.feature_view import build_feature_table


def main() -> int:
    app = QApplication(sys.argv)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (4, 256)) + np.sin(2 * np.pi * 10 * np.arange(256) / 128)[None, :]
    names = ["C3", "C4", "Cz", "Pz"]

    bp = band_powers(data, 128.0)
    tf = time_features(data)
    table = build_feature_table(names, bp, tf)

    print("[1] Filas = canales, columnas = bandas + temporales")
    assert table.rowCount() == 4, table.rowCount()
    assert table.columnCount() == len(bp) + len(tf), table.columnCount()
    cols = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert {"delta", "alpha", "rms", "hjorth_activity"} <= set(cols), cols
    print(f"    {table.rowCount()} canales × {table.columnCount()} características")

    print("[2] Las celdas tienen valor y color")
    item = table.item(0, cols.index("alpha"))
    assert item is not None and item.text(), "celda vacía"
    assert item.background().color().isValid(), "sin color de fondo"
    print(f"    alpha de {names[0]} = {item.text()}")

    print("\nVISOR DE CARACTERÍSTICAS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
