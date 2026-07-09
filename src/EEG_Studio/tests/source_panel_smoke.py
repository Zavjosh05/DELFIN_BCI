"""Panel de fuentes: ordenar (alfabético/fecha/propio) e indicadores de contenido.

Offscreen. Comprueba el reordenado persistente, los modos de orden de la vista y el
indicador de segmentos/marcadores por archivo.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.config import RECORDINGS_DIR  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.core.recording import Recording  # noqa: E402
from eeg_studio.ui import main_window as mw  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402

_ROLE_ID = Qt.ItemDataRole.UserRole
_ROLE_MARK = mw._MARK_COLOR_ROLE          # punto indicador (delegate)


def _csv(path):
    rec = CSVRecorder(path, 14, 128.0)
    rec.write(np.zeros((14, 40)))
    rec.close()


def _rec_with_events(path, n_events):
    data = np.zeros((14, 40))
    events = [{"sample": 5 * (i + 1), "id": str(i + 1)} for i in range(n_events)]
    return Recording(source_path=path, channel_names=[f"Ch{i}" for i in range(14)],
                     data=data, time=np.arange(40) / 128.0, sample_rate=128.0,
                     events=events)


def _ids(win):
    lst = win.sources_list
    return [lst.item(i).data(_ROLE_ID) for i in range(lst.count())]


def _pump(app, win, timeout=3.0):
    t0 = time.time()
    while win._scanning_markers and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    root = tempfile.mkdtemp()
    win.project = Project.create(root, "orden")
    proj = win.project

    paths = {}
    for name in ("Beta", "alfa", "Gamma"):            # orden de creación
        p = os.path.join(proj.path, RECORDINGS_DIR, f"{name}.csv")
        _csv(p)
        paths[name] = p
    # Fechas de modificación controladas: Gamma < Beta < alfa
    base = time.time() - 1000
    os.utime(paths["Gamma"], (base, base))
    os.utime(paths["Beta"], (base + 100, base + 100))
    os.utime(paths["alfa"], (base + 200, base + 200))

    sid_beta = proj.add_source(paths["Beta"], alias="Beta")["id"]
    sid_alfa = proj.add_source(paths["alfa"], alias="alfa")["id"]
    sid_gamma = proj.add_source(paths["Gamma"], alias="Gamma")["id"]

    # Inyecta marcadores en «alfa» (2) ANTES del primer escaneo.
    proj._recordings[sid_alfa] = _rec_with_events(paths["alfa"], 2)

    win._source_sort = "custom"
    win.refresh_all()

    print("[1] Cada item guarda su source_id; orden propio = orden del proyecto")
    assert _ids(win) == [sid_beta, sid_alfa, sid_gamma], _ids(win)
    assert win.sources_list.item(0).text() == "Beta"

    print("[2] Orden alfabético (A→Z, sin distinguir mayúsculas)")
    win._source_sort = "alpha"; win._refresh_sources()
    assert [win.sources_list.item(i).text() for i in range(3)] == ["alfa", "Beta", "Gamma"]

    print("[3] Orden por última modificación (Gamma < Beta < alfa)")
    win._source_sort = "modified"; win._refresh_sources()
    assert _ids(win) == [sid_gamma, sid_beta, sid_alfa], _ids(win)

    print("[4] Orden por fecha de creación no rompe y lista las 3")
    win._source_sort = "created"; win._refresh_sources()
    assert set(_ids(win)) == {sid_beta, sid_alfa, sid_gamma}

    print("[5] Reordenar (orden propio) persiste al reabrir el proyecto")
    assert proj.reorder_sources([sid_gamma, sid_beta, sid_alfa]) is True
    assert [s["id"] for s in proj.sources] == [sid_gamma, sid_beta, sid_alfa]
    assert proj.reorder_sources([sid_gamma, sid_beta, sid_alfa]) is False  # sin cambios
    proj.save()
    reopened = Project.open(proj.path)
    assert [s["id"] for s in reopened.sources] == [sid_gamma, sid_beta, sid_alfa]
    proj.undo()                                        # el reordenado es reversible
    assert [s["id"] for s in proj.sources] == [sid_beta, sid_alfa, sid_gamma]

    print("[6] Indicador de segmentos: punto (color) en la fuente con segmentos")
    proj.add_segment(sid_beta, 0, 10, "clase_1")
    win._source_sort = "custom"; win._refresh_sources()
    it_beta = win._item_for_sid(sid_beta)
    col_seg = it_beta.data(_ROLE_MARK)
    assert isinstance(col_seg, QColor) and col_seg == mw.COLOR_HAS_SEGMENTS, col_seg
    it_gamma = win._item_for_sid(sid_gamma)
    assert it_gamma.data(_ROLE_MARK) is None                     # sin nada, sin punto

    print("[7] Escaneo de marcadores (2 º plano): «alfa» tiene 2 → punto ámbar")
    _pump(app, win)
    assert win._src_event_counts.get(sid_alfa) == 2, win._src_event_counts
    win._update_source_indicators()
    it_alfa = win._item_for_sid(sid_alfa)
    col_mark = it_alfa.data(_ROLE_MARK)
    assert isinstance(col_mark, QColor) and col_mark == mw.COLOR_HAS_MARKERS, col_mark
    assert col_mark != col_seg                                   # distinto a segmentos

    print("[8] El renombrado sigue funcionando aunque la vista esté ordenada")
    win._source_sort = "alpha"; win._refresh_sources()
    it = win._item_for_sid(sid_gamma)
    it.setText("Zeta")                                  # dispara itemChanged → rename
    assert proj.get_source(sid_gamma)["alias"] == "Zeta", proj.get_source(sid_gamma)

    _pump(app, win)
    win.acq_panel.shutdown()
    print("\nPANEL DE FUENTES (ORDEN + INDICADORES) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
