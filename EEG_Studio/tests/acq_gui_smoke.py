"""Prueba de humo de la adquisición integrada en la GUI (offscreen, sin hardware).

Conecta la fuente simulada a través del panel, deja correr el QTimer real,
graba a CSV del proyecto y recarga el archivo. Evita diálogos modales.
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

from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core.csv_loader import load_recording  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _pump(app, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    win.ask_add_recording = lambda p: False  # evita el diálogo modal en el test

    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "acq")
    win.refresh_all()

    panel = win.acq_panel
    print("[1] Conectando fuente simulada vía el panel")
    panel.toggle_connection()
    _pump(app, 1.2)
    assert panel._configured, "el visor en vivo no se configuró"
    assert panel._n_samples > 0, "no llegaron muestras"
    print(f"    muestras recibidas: {panel._n_samples}")

    print("[2] Grabando ~0.6 s a CSV del proyecto")
    panel._start_recording()
    _pump(app, 0.6)
    assert panel.recorder is not None and panel.recorder.n_samples > 0, "no se grabó"
    rec_path = panel._rec_path
    print(f"    grabadas: {panel.recorder.n_samples} muestras")

    print("[3] Deteniendo y desconectando")
    panel._stop_recording()
    panel.toggle_connection()
    assert panel.source is None, "la fuente no se detuvo"

    print("[4] Recargando la grabación con el loader")
    loaded = load_recording(rec_path)
    print(f"    canales={loaded.n_channels} muestras={loaded.n_samples} fs={loaded.sample_rate}")
    assert loaded.n_channels == 14, "nº de canales inesperado"

    win.acq_panel.shutdown()
    print("\nADQUISICIÓN GUI OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
