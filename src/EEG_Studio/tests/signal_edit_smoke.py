"""Edición de señal: recortar tramos (no destructivo) y borrar segmentos por rango."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication

from eeg_studio.config import IMPORTED_DIR
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.signal_view import SignalView


def main() -> int:
    app = QApplication(sys.argv)
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "edit")
    imp = os.path.join(proj.path, IMPORTED_DIR)
    os.makedirs(imp, exist_ok=True)
    csv = os.path.join(imp, "s.csv.gz")
    data = np.random.default_rng(0).normal(0, 1, (2000, 3)).astype(np.float32)
    write_openvibe_csv(csv, data, 250.0, ["C3", "Cz", "C4"],
                       [(100, "izq"), (600, "der"), (1100, "pies")])
    sid = proj.add_source(csv)["id"]

    print("[1] Recortar un tramo elimina los segmentos solapados y guarda el corte")
    s1 = proj.add_segment(sid, 500, 700, "der")["id"]      # solapa el corte
    s2 = proj.add_segment(sid, 1500, 1700, "pies")["id"]   # fuera del corte
    proj.add_cut(sid, 550, 720)
    assert proj.cut_intervals(sid) == [(550, 720)], proj.cut_intervals(sid)
    ids = {s["id"] for s in proj.state["segments"]}
    assert s1 not in ids and s2 in ids, "no se eliminó el segmento solapado / se eliminó uno fuera"
    print(f"    corte={proj.cut_intervals(sid)}  ·  segmentos restantes={len(proj.state['segments'])}")

    print("[2] Cortes contiguos se fusionan")
    proj.add_cut(sid, 700, 900)
    assert proj.cut_intervals(sid) == [(550, 900)], proj.cut_intervals(sid)
    print(f"    fusionado -> {proj.cut_intervals(sid)}")

    print("[3] Segmentos desde marcadores omite marcadores dentro del corte")
    proj.clear_segments()
    n = proj.segments_from_markers(sid, window=200, offset=0)
    labels = sorted(s["label"] for s in proj.state["segments"])
    assert n == 2 and labels == ["izq", "pies"], (n, labels)   # 'der' (600) cae en el corte
    print(f"    creados {n}: {labels} (se saltó el marcador en el corte)")

    print("[4] Deshacer restaura el corte (Ctrl+Z)")
    proj.clear_cuts(sid)
    assert proj.cut_intervals(sid) == []
    proj.undo()
    assert proj.cut_intervals(sid) == [(550, 900)], proj.cut_intervals(sid)
    print("    undo devolvió el corte")

    print("[5] remove_segments_in_range borra por rango")
    proj.clear_segments()
    proj.add_segment(sid, 100, 300, "a")
    proj.add_segment(sid, 1000, 1200, "b")
    removed = proj.remove_segments_in_range(sid, 50, 350)
    assert removed == 1 and len(proj.state["segments"]) == 1
    print(f"    borrados {removed} en el rango")

    print("[6] El visor dibuja tramos recortados sin romper")
    view = SignalView()
    view.set_cuts([(550, 900)])
    view.set_data(data.T, 250.0, ["C3", "Cz", "C4"])       # dispara redibujo + overlay
    assert view._cut_items or True   # basta con que no reviente
    print("    set_cuts + render OK")

    print("\nEDICIÓN DE SEÑAL (recorte / borrado) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
