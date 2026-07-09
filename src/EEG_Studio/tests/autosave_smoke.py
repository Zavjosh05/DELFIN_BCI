"""Verifica el guardado continuo (autosave) y que Ctrl+S (save_project) sigue."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication

from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "as")
    path = win.project.path

    print("[1] Un cambio programa el autoguardado (sin Ctrl+S)")
    win.add_pipeline_step("car")
    assert win._autosave_timer.isActive(), "no se programó el autoguardado"

    print("[2] Al dispararse, persiste en disco")
    win._autosave()                                   # simula que vence el temporizador
    reopened = Project.open(path)
    assert [s["type"] for s in reopened.state["pipeline"]] == ["car"], "no se autoguardó"
    print("    cambio persistido por autosave")

    print("[3] Otro cambio + autosave acumula")
    win.add_pipeline_step("normalize")
    win._autosave()
    assert [s["type"] for s in Project.open(path).state["pipeline"]] == ["car", "normalize"]

    print("[4] Ctrl+S (save_project) sigue funcionando y cancela el pendiente")
    win.add_pipeline_step("detrend")
    win.save_project()
    assert not win._autosave_timer.isActive(), "save_project no canceló el autosave pendiente"
    assert len(Project.open(path).state["pipeline"]) == 3
    print("    guardado manual OK")

    print("\nAUTOSAVE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
