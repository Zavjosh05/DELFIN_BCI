"""Blindaje de grabación: archivo lateral de marcas, pausa y controles. Offscreen."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.acquisition.simulated import SimulatedSource  # noqa: E402
from eeg_studio.core import marks_sidecar  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()

    print("[1] marks_sidecar: escribir/leer/borrar (round-trip)")
    csv = os.path.join(tmp, "x.csv")
    open(csv, "w").close()
    marks_sidecar.write_marks(csv, [(10, 50, "a"), (60, 120, "b")], 128.0)
    assert os.path.isfile(marks_sidecar.sidecar_path(csv))
    assert marks_sidecar.read_marks(csv) == [(10, 50, "a"), (60, 120, "b")]
    marks_sidecar.remove_marks(csv)
    assert not os.path.isfile(marks_sidecar.sidecar_path(csv))
    assert marks_sidecar.read_marks(csv) == []          # sin lateral → vacío
    print("    write → read → remove OK")

    win = MainWindow()
    win.project = Project.create(tmp, "ctl")
    panel = win.acq_panel
    panel.source = SimulatedSource()
    panel._configured = True

    print("[2] Al grabar se crea el lateral; cada marca se vuelca al instante")
    panel.name_edit.setText("prueba_ctl")
    panel._start_recording()
    assert panel.recorder is not None
    rp = panel._rec_path
    assert os.path.isfile(marks_sidecar.sidecar_path(rp)), "no se creó el lateral al iniciar"
    panel.marker_edit.setText("mano_izq")
    panel.duration_spin.setValue(3)
    panel._add_timed_marker()                            # segmento de duración fija
    segs = marks_sidecar.read_marks(rp)
    assert len(segs) == 1 and segs[0][2] == "mano_izq", segs
    print(f"    lateral con {len(segs)} segmento tras marcar")

    print("[3] Pausa: alterna estado y texto; en pausa no se escribe")
    assert not panel._paused
    panel.toggle_pause()
    assert panel._paused and panel.pause_btn.text().startswith("▶")
    panel.toggle_pause()
    assert not panel._paused and panel.pause_btn.text().startswith("⏸")
    print("    pausar/reanudar OK")

    print("[4] Controles pausar/descartar activos solo mientras se graba")
    panel._update_states()
    assert panel.pause_btn.isEnabled() and panel.discard_btn.isEnabled()

    print("[5] Descartar borra archivo + lateral (lógica de limpieza)")
    path = panel._rec_path
    panel.recorder.close()
    panel.recorder = None
    if os.path.isfile(path):
        os.remove(path)
    marks_sidecar.remove_marks(path)
    assert not os.path.isfile(path) and not os.path.isfile(marks_sidecar.sidecar_path(path))
    panel._update_states()
    assert not panel.pause_btn.isEnabled() and not panel.discard_btn.isEnabled()
    print("    descarte deja todo limpio; botones desactivados fuera de grabación")

    win.acq_panel.shutdown()
    print("\nBLINDAJE DE GRABACIÓN (LATERAL + PAUSA + CONTROLES) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
