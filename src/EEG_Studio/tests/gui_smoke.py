"""Prueba de humo de la interfaz en modo offscreen (sin pantalla).

Instancia la ventana, carga un proyecto y una fuente, aplica un pipeline y
fuerza el refresco del visor (que lanza un worker en hilo). No requiere display.

Uso:
    QT_QPA_PLATFORM=offscreen python -m tests.gui_smoke ../EEG/Prueba_001.csv
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main(csv_path: str) -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "gui_smoke")
    win.project.add_source(os.path.abspath(csv_path))
    win.current_source_id = win.project.sources[0]["id"]

    win.project.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    win.project.add_pipeline_step("car")
    win.refresh_all()
    win._update_signal_view()          # lanza worker en hilo
    win._on_segment_requested  # referencia, no se invoca (requiere diálogo)

    # Deja correr el bucle un instante para que el worker termine.
    QTimer.singleShot(1500, app.quit)
    app.exec()

    assert win.sources_list.count() == 1, "la lista de fuentes no se pobló"
    assert win.preproc_panel.steps_list.count() == 2, "el pipeline no se reflejó"
    root = win.changelog_tree.topLevelItem(0)
    assert root is not None and root.childCount() >= 1, "el historial de cambios está vacío"
    print("GUI OK: ventana, visor, paneles y worker funcionan en offscreen.")
    return 0


if __name__ == "__main__":
    from tests import data_dir
    arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        data_dir(), "Prueba_001.csv")
    raise SystemExit(main(arg))
