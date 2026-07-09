"""La barra de pestañas de pipelines refleja el estado y permite eliminar. Offscreen."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "plui")
    win.refresh_all()
    bar = win.preproc_panel.pipeline_bar

    print("[1] Empieza con 1 pipeline en la barra")
    assert bar.count() == 1, bar.count()

    print("[2] Añadir pipelines se refleja en la barra y activa el nuevo")
    win.add_pipeline()
    win.add_pipeline()
    assert bar.count() == 3, bar.count()
    assert bar.currentIndex() == win.project.active_pipeline_index() == 2

    print("[3] Cambiar de pipeline desde la barra actualiza el activo")
    win.preproc_panel._on_pipeline_changed(0)
    assert win.project.active_pipeline_index() == 0
    win.refresh_all()
    assert bar.currentIndex() == 0

    print("[4] Renombrar se refleja en el título de la pestaña")
    win.rename_pipeline(0, "Principal")
    assert bar.tabText(0) == "Principal", bar.tabText(0)

    print("[5] Eliminar un pipeline (botón/☑) lo quita de la barra")
    win.remove_pipeline(2)
    assert bar.count() == 2, bar.count()
    win.remove_pipeline(1)
    assert bar.count() == 1, bar.count()

    print("[6] La casilla de cierre solo aparece con más de un pipeline")
    assert not bar.tabsClosable(), "con un solo pipeline no debe poder cerrarse"
    win.add_pipeline()
    assert bar.tabsClosable(), "con dos o más, sí"

    win.acq_panel.shutdown()
    print("\nBARRA DE PIPELINES (UI) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
