"""Verifica que los marcadores se dibujan sobre la señal (ayuda visual) y que el
etiquetado manual sigue disponible. Offscreen, sin hardware.
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


def _marker_lines(view):
    return [it for it in view.plot.plotItem.items if isinstance(it, pg.InfiniteLine)]


def main() -> int:
    app = QApplication(sys.argv)

    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "marcado.csv")
    rec = CSVRecorder(csv, 14, 128.0)
    rec.add_marker("A"); rec.write(np.zeros((14, 20)))
    rec.add_marker("B"); rec.write(np.ones((14, 20)))
    rec.close()

    win = MainWindow()
    win.project = Project.create(tmp, "mv")
    sid = win.project.add_source(csv)["id"]
    win.current_source_id = sid
    win._update_signal_view()

    print("[1] Los marcadores se dibujan sobre la señal")
    lines = _marker_lines(win.signal_view)
    assert len(lines) == 2, f"se esperaban 2 marcadores dibujados, hay {len(lines)}"
    print(f"    {len(lines)} marcadores dibujados")

    print("[2] El interruptor 'Marcadores' los oculta/muestra")
    win.signal_view.markers_chk.setChecked(False)
    assert len(_marker_lines(win.signal_view)) == 0, "no se ocultaron"
    win.signal_view.markers_chk.setChecked(True)
    assert len(_marker_lines(win.signal_view)) == 2, "no se volvieron a mostrar"
    print("    alternar marcadores OK")

    print("[3] El etiquetado manual sigue disponible")
    assert win.signal_view.add_seg_btn.isEnabled(), "falta el botón de crear segmento"
    s0, s1 = win.signal_view.selection_samples()
    assert s1 > s0, "la selección de región no funciona"
    # La capacidad de añadir segmentos manualmente sigue intacta.
    before = len(win.project.state["segments"])
    win.project.add_segment(sid, s0, s1, "manual")
    assert len(win.project.state["segments"]) == before + 1, "no se pudo etiquetar manualmente"
    print("    selección + etiquetado manual intactos")

    win.acq_panel.shutdown()
    print("\nMARCADORES EN EL VISOR OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
