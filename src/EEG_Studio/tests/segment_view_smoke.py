"""Verifica que los segmentos etiquetados se sombrean sobre la señal (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow
from eeg_studio.ui.signal_view import segment_color

from tests import sample_csv_path


def _texts(view):
    return [it for it in view.plot.plotItem.items if isinstance(it, pg.TextItem)]


def _bands(view):
    return [it for it in view.plot.plotItem.items
            if isinstance(it, pg.LinearRegionItem) and not it.movable]


def main() -> int:
    app = QApplication(sys.argv)
    csv = sample_csv_path()
    assert os.path.isfile(csv), "falta el CSV de ejemplo"

    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "sv")
    sid = win.project.add_source(csv)["id"]
    win.current_source_id = sid

    print("[1] Color estable por clase")
    assert segment_color("A") == segment_color("A"), "el color no es estable"

    print("[2] Dos segmentos etiquetados se sombrean con su clase")
    win.project.add_segment(sid, 0, 256, "A")
    win.project.add_segment(sid, 256, 512, "B")
    win._update_signal_view()
    assert len(_bands(win.signal_view)) == 2, f"bandas: {len(_bands(win.signal_view))}"
    assert len(_texts(win.signal_view)) == 2, f"etiquetas: {len(_texts(win.signal_view))}"
    print(f"    {len(_bands(win.signal_view))} bandas, {len(_texts(win.signal_view))} etiquetas")

    print("[3] Detalle del segmento al pasar el ratón (tooltip)")
    band = _bands(win.signal_view)[0]
    tip = band.toolTip()
    assert "Clase:" in tip and "Rango:" in tip and "Muestras:" in tip, f"tooltip: {tip!r}"
    print(f"    tooltip: {tip.splitlines()[0]} | {tip.splitlines()[1]}")

    print("[4] El interruptor 'Segmentos' las oculta/muestra")
    win.signal_view.segments_chk.setChecked(False)
    assert len(_bands(win.signal_view)) == 0 and len(_texts(win.signal_view)) == 0
    win.signal_view.segments_chk.setChecked(True)
    assert len(_bands(win.signal_view)) == 2
    print("    alternar segmentos OK")

    win.acq_panel.shutdown()
    print("\nSEGMENTOS EN EL VISOR OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
