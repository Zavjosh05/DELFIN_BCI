"""Tema oscuro, barra de herramientas y título de ventana (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication, QToolBar

from eeg_studio.config import APP_NAME
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow
from eeg_studio.ui.theme import BG, apply_dark_theme


def main() -> int:
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    print("[1] La paleta y la hoja de estilo se aplican")
    assert app.palette().window().color().name().lower() == BG.lower(), \
        app.palette().window().color().name()
    assert "QPushButton" in app.styleSheet()
    print(f"    fondo de ventana = {BG}, QSS aplicado")

    win = MainWindow()

    print("[2] Hay barra de herramientas con acciones")
    bars = win.findChildren(QToolBar)
    assert bars and bars[0].actions(), "sin barra de herramientas"
    n_actions = sum(1 for a in bars[0].actions() if not a.isSeparator())
    assert n_actions >= 5, n_actions
    print(f"    {n_actions} acciones en la barra")

    print("[3] Sin proyecto se muestra la pantalla de bienvenida")
    assert win.center_stack.currentWidget() is win.welcome
    assert win.windowTitle() == APP_NAME, win.windowTitle()
    print("    bienvenida visible, título sin proyecto")

    print("[4] El título refleja el proyecto y se ve el área de trabajo")
    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "MiProyecto")
    win.refresh_all()
    assert win.center_stack.currentWidget() is win.center_tabs
    assert "MiProyecto" in win.windowTitle(), win.windowTitle()
    print(f"    título = «{win.windowTitle()}»")

    print("[5] Indicador de cambios sin guardar (●)")
    win.request_autosave()                       # marca sucio (timer no corre sin event loop)
    assert win.windowTitle().startswith("●"), win.windowTitle()
    win.save_project()
    assert not win.windowTitle().startswith("●"), win.windowTitle()
    print("    ● al cambiar, desaparece al guardar")

    win.acq_panel.shutdown()
    print("\nTEMA E INTERFAZ OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
