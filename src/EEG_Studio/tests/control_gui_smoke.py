"""Prueba del modo de control a través de la interfaz (offscreen, sin hardware).

Entrena un modelo, conecta la fuente simulada, inicia el control y comprueba que
clasifica las ventanas en vivo y envía comandos (salida de registro).
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

from PyQt6.QtWidgets import QApplication

from eeg_studio.core import classification, dataset as dataset_mod
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow

from tests import data_dir
EEG_DIR = data_dir()


def _pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def main(csv_path) -> int:
    app = QApplication(sys.argv)
    win = MainWindow()

    print("[1] Proyecto + segmentos + modelo entrenado")
    proj = Project.create(tempfile.mkdtemp(), "ctrl")
    win.project = proj
    sid = proj.add_source(os.path.abspath(csv_path))["id"]
    rec = proj.get_recording(sid)
    for i, ep in enumerate(rec.epoch_ids[:6]):
        a, b = rec.epoch_range(ep)
        proj.add_segment(sid, a, b, label=f"clase_{i % 2}")
    win.dataset = dataset_mod.build_dataset(proj)
    win._register_model(classification.train(win.dataset, "random_forest"))
    win.control_panel.refresh()
    assert win.control_panel.start_btn.isEnabled(), "el control no se habilitó con el modelo"
    assert win.control_panel._cmd_edits, "no se construyó el mapa de comandos por clase"

    print("[2] Conectando la fuente simulada")
    win.acq_panel.toggle_connection()
    _pump(app, 2.5)   # llenar el buffer
    assert win.acq_panel.is_streaming(), "la fuente no transmite"
    assert win.acq_panel.latest_window(128) is not None, "no hay suficientes muestras"

    print("[3] Iniciando el control (salida de registro, K=1)")
    cp = win.control_panel
    cp.window.setValue(128)
    cp.smooth_k.setValue(1)
    cp.sink_combo.setCurrentIndex(0)   # log
    cp._start()
    _pump(app, 1.5)
    assert cp.pred_label.text() != "—", "no se produjo ninguna predicción"
    assert cp.sink is not None and len(cp.sink.history) >= 1, "no se envió ningún comando"
    print(f"    predicción='{cp.pred_label.text()}'  comandos={len(cp.sink.history)}")

    cp._stop()
    win.acq_panel.shutdown()
    print("\nCONTROL (GUI) OK ✓")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EEG_DIR, "Prueba_001.csv")
    raise SystemExit(main(arg))
