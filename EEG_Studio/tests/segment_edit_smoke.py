"""Reetiquetar/eliminar un segmento etiquetado desde el visor (clic derecho). Offscreen."""
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "s.csv")
    rec = CSVRecorder(csv, 14, 128.0)
    rec.write(np.zeros((14, 400)))
    rec.close()

    win = MainWindow()
    win.project = Project.create(tmp, "segedit")
    sid = win.project.add_source(csv)["id"]
    win.current_source_id = sid
    win._open_source_tab(sid)

    seg = win.project.add_segment(sid, 100, 300, "a")
    view = win.signal_view
    win._render_view(sid, view)                    # refleja el segmento (con id) en el visor

    print("[1] El visor conoce el segmento y su id (para el clic derecho)")
    assert any(len(s) >= 4 and s[3] == seg["id"] for s in view._segments), view._segments

    print("[2] _segment_at localiza el segmento por la muestra (dentro/fuera)")
    assert view._segment_at(200) == (seg["id"], "a"), view._segment_at(200)
    assert view._segment_at(50) is None
    assert view._segment_at(350) is None

    print("[3] Eliminar (señal del menú) → el segmento desaparece del proyecto")
    view.delete_segment_requested.emit(seg["id"])
    assert all(s["id"] != seg["id"] for s in win.project.state["segments"])
    print("    segmento eliminado vía clic derecho")

    print("[4] Reetiquetar cambia la clase del segmento (núcleo)")
    seg2 = win.project.add_segment(sid, 100, 300, "a")
    win.project.relabel_segment(seg2["id"], "b")
    lab = next(s["label"] for s in win.project.state["segments"] if s["id"] == seg2["id"])
    assert lab == "b", lab
    # tras reetiquetar, el visor lo refleja con la nueva etiqueta
    win._render_view(sid, view)
    assert view._segment_at(200) == (seg2["id"], "b"), view._segment_at(200)
    print("    reetiquetado a «b» reflejado en el visor")

    print("[5] El segmento más específico gana si hay solapados")
    big = win.project.add_segment(sid, 0, 400, "grande")
    win._render_view(sid, view)
    # en la muestra 200 hay dos (0-400 y 100-300); gana el más pequeño (100-300)
    assert view._segment_at(200) == (seg2["id"], "b"), view._segment_at(200)
    # en la muestra 20 solo está el grande
    assert view._segment_at(20) == (big["id"], "grande"), view._segment_at(20)
    print("    prioridad al segmento más específico OK")

    print("\nEDITAR SEGMENTOS DESDE EL VISOR (CLIC DERECHO) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
