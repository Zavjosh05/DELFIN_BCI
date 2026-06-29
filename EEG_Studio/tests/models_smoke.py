"""Verifica el registro de varios modelos por proyecto, métricas, activar,
exportar/importar, eliminar y persistencia (offscreen)."""
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

from eeg_studio.core import classification
from eeg_studio.core.dataset import Dataset
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def _ds(n=40, d=6):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(3, 1, (n // 2, d))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X, y, [f"f{i}" for i in range(d)], [str(i) for i in range(n)])


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "m")
    win.dataset = _ds()

    print("[1] Entrenar y registrar varios modelos")
    win._register_model(classification.train(win.dataset, "random_forest"))
    win._register_model(classification.train(win.dataset, "lda"))
    assert set(win.models) == {"random_forest_1", "lda_1"}, list(win.models)
    assert win.active_model_name == "lda_1", win.active_model_name
    assert win.clf_panel.models_list.count() == 2
    assert win.control_panel.model_combo.count() == 2
    print(f"    modelos={list(win.models)} activo={win.active_model_name}")

    print("[2] Métricas disponibles (matriz de confusión)")
    r = win.models["random_forest_1"]
    assert r.metrics is not None and "confusion" in r.metrics
    print(f"    rf acc={r.metrics['accuracy']:.3f}")

    print("[3] Activar otro modelo")
    win.activate_model("random_forest_1")
    assert win.model is win.models["random_forest_1"]
    print(f"    activo={win.active_model_name}")

    print("[4] Exportar e importar")
    exp = os.path.join(tempfile.mkdtemp(), "mi_modelo.joblib")
    classification.save_model_to(win.models["lda_1"], exp)
    loaded = classification.load_model(exp)
    win._register_model(loaded, name="importado")
    assert "importado" in win.models and win.models["importado"].metrics is not None
    print("    exportar/importar OK")

    print("[5] Eliminar un modelo (y su archivo)")
    path = os.path.join(win.project.path, "models", "importado.joblib")
    assert os.path.isfile(path)
    win.remove_model("importado")
    assert "importado" not in win.models and not os.path.isfile(path)
    print("    eliminar OK")

    print("[6] Persistencia: recargar modelos del proyecto")
    win._load_project_models()
    assert set(win.models) == {"random_forest_1", "lda_1"}, list(win.models)
    print(f"    recargados={list(win.models)}")

    win.control_panel.shutdown()
    print("\nREGISTRO DE MODELOS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
