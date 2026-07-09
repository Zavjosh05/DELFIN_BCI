"""Verifica la legibilidad del visor con muchos marcadores (culling por viewport).

Con muchos marcadores no se dibujan todos de golpe: solo los visibles en el rango,
y la grabación larga arranca con una ventana acotada (no los ~minutos enteros).
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from eeg_studio.acquisition import CSVRecorder
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "muchos.csv")
    # 100 marcadores cada 300 muestras (1.2 s) a 250 Hz -> grabación de 120 s.
    rec = CSVRecorder(csv, 14, 250.0)
    for i in range(100):
        rec.add_marker(f"c{i % 4}")
        rec.write(np.zeros((14, 300)))
    rec.close()

    win = MainWindow()
    win.project = Project.create(tmp, "cull")
    win.current_source_id = win.project.add_source(csv)["id"]
    win._update_signal_view()
    sv = win.signal_view

    def n_lines():
        return len([it for it in sv.plot.plotItem.items if isinstance(it, pg.InfiniteLine)])

    print("[1] Grabación larga (120 s, 100 marcadores): ventana inicial acotada")
    initial = n_lines()
    assert initial < 100, f"se dibujaron todos ({initial}); no se acotó la ventana"
    print(f"    dibujados al abrir: {initial} de 100")

    print("[2] Zoom a 4 s: solo los marcadores de ese tramo")
    sv.plot.setXRange(0, 4)
    sv._redraw_overlay()
    assert n_lines() == 4, f"esperaba 4 marcadores en [0,4]s, hay {n_lines()}"
    print(f"    en [0,4]s: {n_lines()} marcadores (con etiqueta)")

    print("[3] Ver todo: se dibujan los 100")
    sv.plot.setXRange(0, 120)
    sv._redraw_overlay()
    assert n_lines() == 100, f"esperaba 100, hay {n_lines()}"
    print(f"    vista completa: {n_lines()} marcadores (atenuados, sin etiqueta)")

    win.acq_panel.shutdown()
    print("\nLEGIBILIDAD DE MARCADORES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
