"""Renombrar señales (CSV) desde la lista: alias + archivo interno. Offscreen."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.config import RECORDINGS_DIR  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _csv(path):
    rec = CSVRecorder(path, 14, 128.0)
    rec.write(np.zeros((14, 40)))
    rec.close()


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    root = tempfile.mkdtemp()
    proj = Project.create(root, "rename")

    internal = os.path.join(proj.path, RECORDINGS_DIR, "captura.csv")
    _csv(internal)
    sid_int = proj.add_source(internal, alias="captura")["id"]

    ext_dir = tempfile.mkdtemp()
    external = os.path.join(ext_dir, "externa.csv")
    _csv(external)
    sid_ext = proj.add_source(external, alias="externa")["id"]

    print("[1] Renombrar interna: cambia el alias Y el archivo en disco")
    old_int = proj.get_source(sid_int)["path"]
    assert proj.rename_source(sid_int, "Prueba Uno") is True
    s = proj.get_source(sid_int)
    assert s["alias"] == "Prueba Uno", s["alias"]
    assert os.path.basename(s["path"]) == "Prueba_Uno.csv", s["path"]
    assert os.path.isfile(s["path"]) and not os.path.isfile(old_int)
    assert proj.get_recording(sid_int).data.shape[0] == 14      # sigue cargando
    print(f"    captura.csv -> {os.path.basename(s['path'])}")

    print("[2] Renombrar externa: cambia el alias pero NO toca el archivo de origen")
    old_ext = proj.get_source(sid_ext)["path"]
    assert proj.rename_source(sid_ext, "Externa Renombrada") is True
    se = proj.get_source(sid_ext)
    assert se["alias"] == "Externa Renombrada"
    assert se["path"] == old_ext and os.path.isfile(old_ext)    # origen intacto
    print("    alias cambiado, archivo de origen intacto")

    print("[3] Nombre vacío o igual → no cambia nada")
    assert proj.rename_source(sid_int, "   ") is False
    assert proj.rename_source(sid_int, "Prueba Uno") is False

    print("[4] Colisión de archivo → sufijo _2")
    other = os.path.join(proj.path, RECORDINGS_DIR, "otra.csv")
    _csv(other)
    sid_o = proj.add_source(other, alias="otra")["id"]
    proj.rename_source(sid_o, "Prueba Uno")                     # ya existe Prueba_Uno.csv
    assert os.path.basename(proj.get_source(sid_o)["path"]) == "Prueba_Uno_2.csv", \
        proj.get_source(sid_o)["path"]
    print("    Prueba_Uno.csv ocupado -> Prueba_Uno_2.csv")

    print("[5] Deshacer restaura el alias (undo/redo del renombrado)")
    proj.undo()
    assert proj.get_source(sid_o)["alias"] == "otra"
    proj.redo()
    assert proj.get_source(sid_o)["alias"] == "Prueba Uno"

    print("[6] UI: edición en el sitio de la lista dispara el renombrado")
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "ui")
    csv2 = os.path.join(win.project.path, RECORDINGS_DIR, "grab.csv")
    _csv(csv2)
    win.project.add_source(csv2, alias="grab")
    win.refresh_all()
    item = win.sources_list.item(0)
    item.setText("Renombrado En Sitio")           # dispara itemChanged -> rename
    assert win.project.sources[0]["alias"] == "Renombrado En Sitio", win.project.sources[0]
    assert Qt.ItemFlag.ItemIsEditable & win.sources_list.item(0).flags()   # editable
    win.acq_panel.shutdown()
    print("    edición en la lista renombra la señal")

    print("\nRENOMBRAR SEÑALES (CSV) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
