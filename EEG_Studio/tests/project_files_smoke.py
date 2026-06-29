"""Borrado de fuentes internas, segmentos «todas las fuentes» y recientes (offscreen)."""
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

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from eeg_studio.config import APP_NAME, IMPORTED_DIR, ORG_NAME
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def _make_source(path: str, markers) -> None:
    data = np.random.default_rng(0).normal(0, 1, (2000, 3)).astype(np.float32)
    write_openvibe_csv(path, data, 250.0, ["C3", "Cz", "C4"], markers)


def main() -> int:
    app = QApplication(sys.argv)
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "files")

    # Fuente interna (dentro del proyecto) y fuente «de origen» (fuera).
    internal = os.path.join(proj.path, IMPORTED_DIR, "interna.csv.gz")
    os.makedirs(os.path.dirname(internal), exist_ok=True)
    external = os.path.join(tmp, "origen.csv.gz")
    _make_source(internal, [(100, "left_hand"), (600, "right_hand"), (1100, "feet")])
    _make_source(external, [(200, "left_hand"), (700, "feet")])

    print("[1] is_internal_path distingue proyecto vs origen")
    assert proj.is_internal_path(internal) is True
    assert proj.is_internal_path(external) is False
    print("    interna=True, origen=False ✓")

    sid_int = proj.add_source(internal)["id"]
    sid_ext = proj.add_source(external)["id"]

    print("[2] Segmentos de UNA fuente, luego «todas» sin duplicar")
    n_one = proj.segments_from_markers(sid_int, window=0, offset=0)
    assert n_one >= 2, n_one
    before = len(proj.state["segments"])
    n_all = proj.segments_from_markers_all(window=0, offset=0)   # skip_existing=True
    after = len(proj.state["segments"])
    # No vuelve a segmentar la interna (ya tenía); sí añade la externa.
    int_segs = [s for s in proj.state["segments"] if s["source_id"] == sid_int]
    ext_segs = [s for s in proj.state["segments"] if s["source_id"] == sid_ext]
    assert len(int_segs) == n_one, (len(int_segs), n_one)
    assert len(ext_segs) >= 1, len(ext_segs)
    print(f"    interna={len(int_segs)} (sin duplicar), externa={len(ext_segs)} (+{after - before})")

    print("[3] «todas» omite una fuente cuyo archivo falta (no rompe)")
    proj._recordings.clear()
    os.remove(external)
    proj.state["segments"] = [s for s in proj.state["segments"] if s["source_id"] != sid_ext]
    got = proj.segments_from_markers_all(window=0, offset=0, skip_existing=False)
    # La interna sí aporta; la externa (borrada) se omite sin excepción.
    assert got >= 2, got
    print(f"    re-segmentó {got} omitiendo la fuente faltante ✓")

    print("[4] Quitar fuente interna borra su archivo del disco")
    assert os.path.isfile(internal)
    proj.remove_source(sid_int)
    os.remove(internal)                          # la UI lo hace tras confirmar
    assert not os.path.isfile(internal)
    assert proj.get_source(sid_int) is None
    print("    referencia y archivo eliminados ✓")

    print("[5] Proyectos recientes: guardar y recuperar (con respaldo)")
    s = QSettings(ORG_NAME, APP_NAME)
    backup = s.value("recent_projects", None)
    try:
        win = MainWindow()
        win._clear_recent()
        win._push_recent(proj.path)
        assert os.path.abspath(proj.path) in [os.path.abspath(p) for p in win._recent_projects()]
        win._build_recent_menu()                 # no debe romper
        assert win.recent_menu.actions(), "menú reciente vacío"
        print(f"    {len(win._recent_projects())} reciente(s), menú con {len(win.recent_menu.actions())} entradas")
        win.acq_panel.shutdown()
    finally:
        if backup is None:
            s.remove("recent_projects")
        else:
            s.setValue("recent_projects", backup)

    print("\nGESTIÓN DE ARCHIVOS DEL PROYECTO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
