"""Blindaje de la grabación: captura en el hilo productor (independiente de la
GUI), volcado a disco (fsync) y medidor de batería. Offscreen."""
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

from eeg_studio.acquisition.base import StreamSource  # noqa: E402
from eeg_studio.acquisition.emotiv import battery_from_frame  # noqa: E402
from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402


class _FixedSource(StreamSource):
    """Fuente que emite N bloques y termina (para probar el tap sin GUI)."""
    display_name = "fixed"

    def __init__(self, n_emit):
        super().__init__([f"Ch{i}" for i in range(4)], 128.0)
        self._n_emit = n_emit

    def _run(self):
        for _ in range(self._n_emit):
            if not self._running.is_set():
                break
            self._emit(np.ones((self.n_channels, 1)))
            time.sleep(0.001)


def main() -> int:
    root = tempfile.mkdtemp()

    print("[1] La grabación captura TODO vía el tap, sin llamar nunca a read() (GUI)")
    path = os.path.join(root, "tap.csv")
    rec = CSVRecorder(path, 4, 128.0)
    src = _FixedSource(60)
    src.set_tap(rec.write)                    # graba en el hilo productor
    src.start()
    for _ in range(200):                      # espera a que termine (sin leer la cola)
        if not src.is_running():
            break
        time.sleep(0.01)
    src.stop()
    src.set_tap(None)
    rec.close()
    assert rec.n_samples == 60, rec.n_samples  # capturó las 60 aunque nadie hizo read()
    with open(path, encoding="utf-8") as fh:
        rows = [ln for ln in fh.read().splitlines() if ln]
    assert len(rows) == 61, len(rows)          # cabecera + 60 muestras
    print(f"    {rec.n_samples} muestras grabadas por el productor (GUI nunca consumió)")

    print("[2] Volcado a disco periódico (fsync): los datos llegan antes de cerrar")
    p2 = os.path.join(root, "sync.csv")
    rec2 = CSVRecorder(p2, 2, 10.0)            # sync cada ~10 muestras
    header_size = os.path.getsize(p2)          # cabecera ya en disco
    for _ in range(10):
        rec2.write(np.zeros((2, 1)))           # al llegar a 10, hace flush+fsync
    assert os.path.getsize(p2) > header_size, "los datos no se volcaron a disco antes de cerrar"
    rec2.close()
    print("    los datos están en disco sin depender del cierre")

    print("[3] Escribir tras cerrar no rompe (carrera tap/cierre)")
    assert rec2.write(np.zeros((2, 1))) == 0

    print("[4] Batería decodificada del frame del Emotiv (byte contador > 127)")
    frame_full = bytes([248] + [0] * 31)       # 248 -> 100%
    frame_low = bytes([234] + [0] * 31)        # 234 -> 20%
    frame_data = bytes([42] + [0] * 31)        # contador normal -> None
    assert battery_from_frame(frame_full) == 100
    assert battery_from_frame(frame_low) == 20
    assert battery_from_frame(frame_data) is None

    print("[5] Medidor de batería en el panel + umbral configurable + aviso")
    from PyQt6.QtWidgets import QApplication
    from eeg_studio.core.project import Project
    from eeg_studio.ui.main_window import MainWindow
    app = QApplication(sys.argv)               # noqa: F841
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "bat")
    win.warn = lambda *a, **k: None            # evita el diálogo modal en el test
    panel = win.acq_panel

    class _Src:
        battery = 85
        sample_rate = 128.0
    panel.source = _Src()

    panel.battery_thresh.setValue(70)
    panel._update_battery()
    assert panel.battery_row.isVisibleTo(panel) or panel.battery_row.isVisible()
    assert "85" in panel.battery_label.text() and not panel._battery_warned
    panel.source.battery = 60                  # cae por debajo del 70%
    panel._update_battery()
    assert "BAJA" in panel.battery_label.text() and panel._battery_warned
    # sin batería (otras fuentes) → se oculta la fila
    panel.source = None
    panel._update_battery()
    assert not panel.battery_row.isVisible()
    win.acq_panel.shutdown()
    print("    batería mostrada, aviso por debajo del umbral, oculta si no se reporta")

    print("\nBLINDAJE DE GRABACIÓN + BATERÍA OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
