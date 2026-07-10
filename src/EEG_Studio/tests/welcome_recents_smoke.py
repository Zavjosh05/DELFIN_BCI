"""Pantalla de bienvenida: la lista de proyectos recientes permite renombrar
(carpeta .eegproj + nombre interno) y quitar entradas. Offscreen, sin tocar los
ajustes reales del usuario (se respalda y restaura ``recent_projects``)."""
from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import eeg_studio.ui.main_window as mw  # noqa: E402
from eeg_studio.config import PROJECT_EXT, PROJECT_MANIFEST  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    win = MainWindow()

    saved_recent = win._settings().value("recent_projects", [])   # respaldo
    try:
        base = tempfile.mkdtemp()
        pa = Project.create(base, "Alpha").path
        pb = Project.create(base, "Beta").path
        win._settings().setValue("recent_projects", [pa, pb])

        print("[1] La bienvenida lista los recientes con su ruta")
        win._refresh_welcome_recents()
        assert win.welcome_recent.count() == 2, win.welcome_recent.count()
        paths = [win.welcome_recent.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(2)]
        assert os.path.abspath(pa) in [os.path.abspath(p) for p in paths]

        print("[2] Renombrar: mueve la carpeta .eegproj y ajusta el nombre interno")
        mw.QInputDialog.getText = staticmethod(lambda *a, **k: ("Gamma", True))
        win._rename_recent_project(pa)
        new_path = os.path.join(base, "Gamma" + PROJECT_EXT)
        assert not os.path.exists(pa), "la carpeta vieja debería haberse renombrado"
        assert os.path.isfile(os.path.join(new_path, PROJECT_MANIFEST)), "falta project.json"
        with open(os.path.join(new_path, PROJECT_MANIFEST), encoding="utf-8") as fh:
            assert json.load(fh)["name"] == "Gamma", "el nombre interno no se actualizó"
        recent = win._recent_projects()
        assert os.path.abspath(new_path) in [os.path.abspath(p) for p in recent]
        assert os.path.abspath(pa) not in [os.path.abspath(p) for p in recent]
        print(f"    Alpha -> Gamma ({os.path.basename(new_path)}) y recientes actualizado")

        print("[3] Renombrar a un nombre ya existente NO pisa la otra carpeta")
        # Beta ya existe; intentar renombrar Gamma a «Beta» debe rechazarse.
        warned = {"n": 0}
        mw.QMessageBox.warning = staticmethod(lambda *a, **k: warned.update(n=warned["n"] + 1))
        mw.QInputDialog.getText = staticmethod(lambda *a, **k: ("Beta", True))
        win._rename_recent_project(new_path)
        assert warned["n"] == 1, "no avisó del conflicto de nombre"
        assert os.path.isdir(new_path), "no debió mover Gamma"
        assert os.path.isdir(pb), "no debió tocar Beta"

        print("[4] Quitar de la lista solo olvida el reciente (no borra del disco)")
        win._forget_recent(pb)
        win._refresh_welcome_recents()
        recent = win._recent_projects()
        assert os.path.abspath(pb) not in [os.path.abspath(p) for p in recent]
        assert os.path.isdir(pb), "quitar de recientes NO debe borrar la carpeta"
        assert win.welcome_recent.count() == 1, win.welcome_recent.count()
    finally:
        win._settings().setValue("recent_projects", saved_recent)   # restaura ajustes reales

    print("\nRECIENTES: RENOMBRAR + QUITAR OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
