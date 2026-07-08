"""Recuperar grabaciones sueltas (con sus marcas del sidecar) + persistencia. Offscreen."""
from __future__ import annotations

import gzip
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.config import RECORDINGS_DIR  # noqa: E402
from eeg_studio.core import marks_sidecar  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _csv(path):
    rec = CSVRecorder(path, 14, 128.0)
    rec.write(np.zeros((14, 400)))
    rec.close()


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()
    win = MainWindow()
    win.project = Project.create(tmp, "orphans")
    rec_dir = os.path.join(win.project.path, RECORDINGS_DIR)

    a = os.path.join(rec_dir, "a.csv"); _csv(a)
    b = os.path.join(rec_dir, "b.csv"); _csv(b)
    c = os.path.join(rec_dir, "c.csv.gz")
    with open(a, "rb") as fi, gzip.open(c, "wb") as fo:
        fo.write(fi.read())
    # 'b' tiene un archivo lateral con 2 segmentos (como si se hubieran marcado).
    marks_sidecar.write_marks(b, [(10, 90, "clase_a"), (150, 300, "clase_b")], 128.0)
    win.project.add_source(a, alias="a")          # solo 'a' está registrada

    print("[1] orphan_recordings detecta las NO registradas (incluye .csv.gz)")
    orphans = win.project.orphan_recordings()
    names = sorted(os.path.basename(p) for p in orphans)
    assert names == ["b.csv", "c.csv.gz"], names
    print(f"    sueltas: {names}")

    print("[2] add_orphan_recordings las añade, RESTAURA sus marcas y GUARDA en disco")
    n = win.add_orphan_recordings(orphans)
    assert n == 2, n
    assert len(win.project.sources) == 3, len(win.project.sources)
    # las marcas de 'b' se recuperaron del sidecar
    labs = sorted(s["label"] for s in win.project.state["segments"])
    assert labs == ["clase_a", "clase_b"], labs
    # persistido en disco sin depender de cerrar: reabrir lo confirma
    reop = Project.open(win.project.path)
    assert len(reop.sources) == 3 and len(reop.state["segments"]) == 2
    assert win.project.orphan_recordings() == []
    print(f"    añadidas {n}, segmentos recuperados {labs}, persistido OK")

    print("[3] Añadir una grabación como fuente PERSISTE de inmediato en disco")
    newc = os.path.join(rec_dir, "nueva.csv"); _csv(newc)
    win.add_recording_as_source(newc, alias="nueva")
    reop2 = Project.open(win.project.path)
    assert any(s["alias"] == "nueva" for s in reop2.sources), "el alta no se guardó en disco"
    assert not win._dirty, "tras guardar inmediato no debe quedar pendiente"
    print(f"    reabierto con {len(reop2.sources)} fuentes (guardado inmediato)")

    win.acq_panel.shutdown()
    print("\nRECUPERAR GRABACIONES SUELTAS (+ MARCAS) + PERSISTENCIA OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
