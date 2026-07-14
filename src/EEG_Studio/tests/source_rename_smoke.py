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

    print("[5] Deshacer/rehacer restauran el alias Y el archivo en disco")
    # Regresión: antes el undo devolvía la ruta anterior pero NO renombraba el
    # archivo, así que la fuente quedaba apuntando a un archivo inexistente.
    renamed_path = proj.get_source(sid_o)["path"]
    proj.undo()
    s_o = proj.get_source(sid_o)
    assert s_o["alias"] == "otra"
    assert os.path.basename(s_o["path"]) == "otra.csv", s_o["path"]
    assert os.path.isfile(s_o["path"]), "deshacer dejó la fuente apuntando a un archivo inexistente"
    assert not os.path.exists(renamed_path), "el archivo debió volver a su nombre anterior"
    proj.redo()
    s_o = proj.get_source(sid_o)
    assert s_o["alias"] == "Prueba Uno"
    assert os.path.isfile(s_o["path"]), "rehacer dejó la fuente rota"
    assert os.path.basename(s_o["path"]) == "Prueba_Uno_2.csv", s_o["path"]
    print("    el alias y el archivo vuelven juntos (undo y redo)")

    print("[5b] Las marcas (archivo lateral) siguen al CSV al renombrar")
    # Regresión: el lateral se llama «<csv>.marks.json»; si no se mueve con el CSV,
    # las marcas de la grabación quedan huérfanas y se pierden.
    from eeg_studio.core import marks_sidecar
    csv_m = os.path.join(proj.path, RECORDINGS_DIR, "con_marcas.csv")
    _csv(csv_m)
    marks_sidecar.write_marks(csv_m, [(0, 10, "arriba")], 128.0)
    sid_m = proj.add_source(csv_m, alias="con marcas")["id"]
    proj.rename_source(sid_m, "Sujeto007")
    p_new = proj.get_source(sid_m)["path"]
    assert os.path.basename(p_new) == "Sujeto007.csv", p_new
    assert marks_sidecar.read_marks(p_new) == [(0, 10, "arriba")], "se perdieron las marcas"
    assert not os.path.exists(marks_sidecar.sidecar_path(csv_m)), "quedó un lateral huérfano"
    proj.undo()                                   # el lateral también vuelve
    assert marks_sidecar.read_marks(proj.get_source(sid_m)["path"]) == [(0, 10, "arriba")]
    proj.redo()
    assert marks_sidecar.read_marks(proj.get_source(sid_m)["path"]) == [(0, 10, "arriba")]
    print("    <csv>.marks.json se mueve con el CSV (y vuelve al deshacer)")

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
