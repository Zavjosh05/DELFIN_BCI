"""Prueba de la red neuronal a través de la interfaz (offscreen, sin hardware).

Configura una red desde el panel, entrena en segundo plano (MLP y CNN) y predice
la selección actual. Usa el CSV real de ejemplo.
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

from eeg_studio.core import dataset as dataset_mod  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _wait(app, cond, timeout=120.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.05)
    return False


def _select_clf(panel, key: str) -> None:
    idx = panel.clf_combo.findData(key)
    panel.clf_combo.setCurrentIndex(idx)


def main(csv_path: str) -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    proj = Project.create(tempfile.mkdtemp(), "nn")
    win.project = proj
    src = proj.add_source(os.path.abspath(csv_path))
    win.current_source_id = src["id"]
    rec = proj.get_recording(src["id"])

    # 6 segmentos en 2 clases a partir de las épocas.
    for i, ep in enumerate(rec.epoch_ids[:6]):
        a, b = rec.epoch_range(ep)
        proj.add_segment(src["id"], a, b, label=f"clase_{i % 2}")
    win.refresh_all()
    win._update_signal_view()

    panel = win.clf_panel

    print("[1] MLP vía panel (características)")
    win.dataset = dataset_mod.build_dataset(proj)   # dataset de características
    _select_clf(panel, "nn_mlp")
    # isHidden() refleja la intención (isVisible depende de la pestaña activa).
    assert not panel.nn_config_widget.isHidden(), "no se mostró la config de la red"
    panel.nn_config_widget.epochs.setValue(20)
    win.model = None
    win.train_model()
    assert _wait(app, lambda: win.model is not None), "el MLP no terminó de entrenar"
    assert win.model.input_kind == "features"
    print(f"    entrenado · {win.model.score_label}={win.model.cv_mean:.3f}")

    print("[2] CNN vía panel (señal cruda)")
    _select_clf(panel, "nn_cnn")
    panel.nn_config_widget.epochs.setValue(15)
    panel.nn_config_widget.window.setValue(256)
    win.model = None
    win.train_model()                                # construye dataset crudo + entrena
    assert _wait(app, lambda: win.model is not None), "la CNN no terminó de entrenar"
    assert win.model.input_kind == "raw"
    print(f"    entrenado · ventana={win.model.nn_config['window_samples']}")

    print("[3] Predecir la selección actual con la CNN")
    win.predict_selection()
    txt = panel.result_label.text()
    print(f"    {txt.splitlines()[0]}")
    assert "Predicción" in txt, "no se obtuvo predicción"

    win.acq_panel.shutdown()
    print("\nNN GUI OK ✓")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "EEG", "Prueba_001.csv"
    )
    raise SystemExit(main(arg))
