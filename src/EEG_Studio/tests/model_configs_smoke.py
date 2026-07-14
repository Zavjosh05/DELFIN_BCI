"""Configuraciones de modelo: definirlas y GUARDARLAS sin entrenar.

Los valores por defecto de cada clasificador son los de siempre; aquí se prueba
que se pueden guardar/cargar/quitar configuraciones con nombre en el proyecto
(persisten, con undo/redo) y que cargarlas o guardarlas NO entrena nada.
Offscreen.
"""
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

import eeg_studio.ui.panels as panels_mod  # noqa: E402
from eeg_studio.core.dataset import Dataset  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _ds(n=40, d=6):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(3, 1, (n // 2, d))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
                   segment_ids=[str(i) for i in range(n)])


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841

    print("[1] El proyecto guarda/actualiza/quita configuraciones (sin entrenar)")
    proj = Project.create(tempfile.mkdtemp(), "cfg")
    assert proj.model_configs() == []
    proj.save_model_config({"name": "mi_rf", "classifier_name": "random_forest",
                            "clf_params": {"n_estimators": 111}})
    assert len(proj.model_configs()) == 1
    proj.save_model_config({"name": "mi_rf", "classifier_name": "random_forest",
                            "clf_params": {"n_estimators": 222}})   # actualiza, no duplica
    assert len(proj.model_configs()) == 1
    assert proj.model_configs()[0]["clf_params"]["n_estimators"] == 222
    proj.undo()                                        # edición no destructiva
    assert proj.model_configs()[0]["clf_params"]["n_estimators"] == 111
    proj.redo()
    proj.remove_model_config("mi_rf")
    assert proj.model_configs() == []
    try:
        proj.save_model_config({"classifier_name": "lda"})          # sin nombre
        assert False, "debería exigir nombre"
    except ValueError:
        pass
    print("    guardar/actualizar/quitar + undo/redo OK")

    print("[2] Persisten al guardar y reabrir el proyecto")
    proj.save_model_config({"name": "lda_shrink", "classifier_name": "lda",
                            "clf_params": {"solver": "lsqr", "shrinkage": "auto"}})
    proj.save()
    re = Project.open(proj.path)
    assert [c["name"] for c in re.model_configs()] == ["lda_shrink"], re.model_configs()
    assert re.model_configs()[0]["clf_params"] == {"solver": "lsqr", "shrinkage": "auto"}
    print("    configuración recuperada tras reabrir")

    print("[3] Un proyecto SIN la clave (versión antigua) sigue abriendo")
    import json
    mpath = os.path.join(re.path, "project.json")
    data = json.load(open(mpath, encoding="utf-8"))
    data["state"].pop("model_configs", None)               # simula proyecto viejo
    json.dump(data, open(mpath, "w", encoding="utf-8"), ensure_ascii=False)
    old = Project.open(re.path)
    assert old.model_configs() == [], old.model_configs()
    print("    proyecto sin la clave -> lista vacía, sin romper")

    print("[4] La interfaz lista solo lo guardado y lo carga SIN entrenar")
    proj2 = Project.create(tempfile.mkdtemp(), "ui")
    proj2.save_model_config({"name": "lda_shrink", "classifier_name": "lda",
                             "clf_params": {"solver": "lsqr", "shrinkage": "auto"}})
    win = MainWindow()
    win.project = proj2
    win.refresh_all()
    panel = win.clf_panel
    panel.clf_combo.setCurrentIndex(panel.clf_combo.findData("lda"))
    names = [panel.cfg_combo.itemText(i) for i in range(panel.cfg_combo.count())]
    # Solo lo guardado + la entrada de valores por defecto (nada "de fábrica").
    assert names == [panels_mod.DEFAULT_CONFIG_NAME, "lda_shrink"], names
    n_models = len(win.models)
    panel.cfg_combo.setCurrentIndex(panel.cfg_combo.findText("lda_shrink"))
    panel.load_selected_config()
    assert panel.classic_params() == {"solver": "lsqr", "shrinkage": "auto"}
    assert len(win.models) == n_models, "cargar una configuración NO debe entrenar"
    # Un clasificador sin configuraciones guardadas: solo los valores por defecto.
    panel.clf_combo.setCurrentIndex(panel.clf_combo.findData("svm"))
    only = [panel.cfg_combo.itemText(i) for i in range(panel.cfg_combo.count())]
    assert only == [panels_mod.DEFAULT_CONFIG_NAME], only
    assert panel.selected_config().get("is_default") is True
    print(f"    {names} · cargada sin entrenar · sin guardadas -> solo por defecto")

    print("[5] Guardar los valores actuales como configuración nueva (sin entrenar)")
    panel.clf_combo.setCurrentIndex(panel.clf_combo.findData("random_forest"))
    panel.rf_estimators.setValue(150)
    panel.rf_min_leaf.setValue(2)

    orig = panels_mod.QInputDialog

    class _FakeInput:
        @staticmethod
        def getText(*a, **k):
            return ("rf_mio", True)
    panels_mod.QInputDialog = _FakeInput
    try:
        panel.save_current_config()
    finally:
        panels_mod.QInputDialog = orig
    saved = {c["name"]: c for c in win.project.model_configs()}
    assert "rf_mio" in saved, list(saved)
    assert saved["rf_mio"]["clf_params"]["n_estimators"] == 150
    assert saved["rf_mio"]["clf_params"]["min_samples_leaf"] == 2
    assert len(win.models) == n_models, "guardar una configuración NO debe entrenar"
    print("    guardada «rf_mio» (n_estimators=150) sin entrenar")

    print("[6] Eliminar la configuración guardada")
    panel.refresh_model_configs()
    panel.cfg_combo.setCurrentIndex(panel.cfg_combo.findText("rf_mio"))
    panel.remove_selected_config()
    assert "rf_mio" not in {c["name"] for c in win.project.model_configs()}
    print("    eliminada")

    print("[7] «Valores por defecto» siempre está y restaura los valores del programa")
    panel.clf_combo.setCurrentIndex(panel.clf_combo.findData("random_forest"))
    names = [panel.cfg_combo.itemText(i) for i in range(panel.cfg_combo.count())]
    assert names == [panels_mod.DEFAULT_CONFIG_NAME], names
    # Los valores por defecto REALES del programa (capturados al construir la UI),
    # no lo que haya ahora en los campos.
    default_rf = panel.default_config_dict("random_forest")["clf_params"]
    panel.rf_estimators.setValue(999)
    panel.rf_min_leaf.setValue(9)
    assert panel.classic_params()["n_estimators"] == 999
    panel.cfg_combo.setCurrentIndex(
        panel.cfg_combo.findText(panels_mod.DEFAULT_CONFIG_NAME))
    panel.load_selected_config()                       # carga «Valores por defecto»
    assert panel.classic_params() == default_rf, panel.classic_params()
    # No es una configuración guardada: no se puede borrar.
    warned = {"n": 0}
    win.warn = lambda *a, **k: warned.update(n=warned["n"] + 1)
    panel.remove_selected_config()
    assert warned["n"] == 1
    print(f"    restaura los valores por defecto (n_estimators={default_rf['n_estimators']}) "
          "y no se puede borrar")

    print("[8] Entrenar TODAS las configuraciones guardadas de una vez")
    import time
    from PyQt6.QtWidgets import QMessageBox
    win.dataset = _ds()
    win.project.save_model_config({"name": "rf_A", "classifier_name": "random_forest",
                                   "clf_params": {"n_estimators": 30}})
    win.project.save_model_config({"name": "lda_B", "classifier_name": "lda",
                                   "clf_params": {"solver": "lsqr", "shrinkage": "auto"}})
    # El proyecto ya tenía «lda_shrink» del paso [4]: se entrenan las TRES.
    expected = {"lda_shrink", "rf_A", "lda_B"}
    orig_q = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        win.train_all_saved_configs()
        end = time.time() + 90
        while time.time() < end and len(win.models) < len(expected):
            app.processEvents(); time.sleep(0.02)
        assert set(win.models) == expected, list(win.models)
        assert win.models["rf_A"].model.named_steps["clf"].n_estimators == 30
        print(f"    entrenados de golpe: {sorted(win.models)}")

        print("[9] Reentrenar TODOS con los datos actuales (p. ej. cambió el dataset)")
        win.dataset = _ds(n=60)                        # dataset nuevo
        before = {k: id(v) for k, v in win.models.items()}
        win.retrain_all_models()
        end = time.time() + 90
        while time.time() < end and any(id(win.models[k]) == before[k] for k in before):
            app.processEvents(); time.sleep(0.02)
        assert set(win.models) == expected, list(win.models)              # mismos nombres
        assert all(id(win.models[k]) != before[k] for k in before), "no se reentrenaron"
        assert win.models["rf_A"].model.named_steps["clf"].n_estimators == 30  # params
        assert win.models["rf_A"].n_samples == 60, win.models["rf_A"].n_samples
    finally:
        QMessageBox.question = orig_q
    print("    mismos nombres, hiperparámetros conservados, entrenados con 60 muestras")

    win.acq_panel.shutdown()
    print("\nCONFIGURACIONES DE MODELO (SIN ENTRENAR) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
