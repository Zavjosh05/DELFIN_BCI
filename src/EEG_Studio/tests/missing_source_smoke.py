"""Verifica que una fuente cuyo archivo ya no existe no rompe la app (offscreen)."""
from __future__ import annotations

import os
import shutil
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

from tests import data_dir
EEG_DIR = data_dir()


def main() -> int:
    app = QApplication(sys.argv)
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "fuente.csv")
    shutil.copy(os.path.join(EEG_DIR, "Prueba_001.csv"), csv)

    win = MainWindow()
    win.project = Project.create(tmp, "miss")
    sid = win.project.add_source(csv)["id"]
    win.current_source_id = sid
    win._update_signal_view()                      # carga normal

    print("[1] Se borra el archivo de la fuente")
    win.project._recordings.clear()                # forzar recarga desde disco
    os.remove(csv)

    print("[2] _update_signal_view no revienta (lo gestiona)")
    win._update_signal_view()                      # no debe lanzar FileNotFoundError
    print("    gestionado sin crash")

    win.acq_panel.shutdown()
    print("\nFUENTE FALTANTE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
