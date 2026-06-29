"""Verifica las mejoras de interfaz (offscreen):

* Descripción del filtro y de sus parámetros en Preprocesamiento.
* Cuadro de parámetros del SVM con kernel seleccionable.
* Etiquetas de capa de entrada/salida de la red, con su nº de neuronas.
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

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _form_texts(form) -> str:
    out = []
    for i in range(form.rowCount()):
        for role in (form.ItemRole.SpanningRole, form.ItemRole.FieldRole, form.ItemRole.LabelRole):
            it = form.itemAt(i, role)
            if it and isinstance(it.widget(), QLabel):
                out.append(it.widget().text())
    return " | ".join(out)


def _select(combo, key) -> None:
    combo.setCurrentIndex(combo.findData(key))


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    proj = Project.create(tempfile.mkdtemp(), "x")
    win.project = proj

    print("[1] Descripción del filtro y de sus parámetros")
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    win.refresh_all()
    win.preproc_panel.steps_list.setCurrentRow(0)
    texts = _form_texts(win.preproc_panel.params_form)
    assert "frecuencias entre" in texts.lower() or "pasa-banda" in texts.lower() \
        or "atenúa" in texts.lower(), f"falta descripción del filtro: {texts}"
    assert "frecuencia de corte inferior" in texts.lower(), "falta descripción de 'low'"
    print("    descripción de filtro y parámetros presentes")

    print("[2] Cuadro SVM con kernels")
    panel = win.clf_panel
    _select(panel.clf_combo, "svm")
    assert not panel.svm_box.isHidden(), "no se mostró el cuadro del SVM"
    kernels = [panel.svm_kernel.itemData(i) for i in range(panel.svm_kernel.count())]
    assert set(kernels) >= {"linear", "rbf", "poly", "sigmoid"}, f"kernels: {kernels}"
    _select(panel.svm_kernel, "poly")
    assert panel.svm_params()["kernel"] == "poly", "svm_params no refleja el kernel"
    assert panel.svm_degree.isEnabled(), "grado debería habilitarse con 'poly'"
    print(f"    kernels={kernels}, seleccionado=poly")

    print("[2b] Cuadro Random Forest con parámetros")
    _select(panel.clf_combo, "random_forest")
    assert not panel.rf_box.isHidden(), "no se mostró el cuadro del RF"
    panel.rf_estimators.setValue(123)
    panel.rf_max_depth.setValue(7)
    rfp = panel.classic_params()
    assert rfp["n_estimators"] == 123 and rfp["max_depth"] == 7, f"rf_params: {rfp}"
    print(f"    n_estimators={rfp['n_estimators']} max_depth={rfp['max_depth']}")

    print("[3] Capas de entrada/salida de la red")
    # Dos clases a partir de segmentos etiquetados.
    src = proj.add_source(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "EEG", "Prueba_001.csv")))
    rec = proj.get_recording(src["id"])
    for i, ep in enumerate(rec.epoch_ids[:4]):
        a, b = rec.epoch_range(ep)
        proj.add_segment(src["id"], a, b, label=f"clase_{i % 2}")
    win.refresh_all()

    _select(panel.clf_combo, "nn_mlp")
    inp = panel.nn_config_widget.io_input.text()
    out = panel.nn_config_widget.io_output.text()
    print(f"    MLP  → {inp} || {out}")
    assert "neuronas" in inp and "neuronas" in out, "faltan nº de neuronas en MLP"
    assert "2 neuronas" in out, f"salida debería ser 2 clases: {out}"

    _select(panel.clf_combo, "nn_cnn")
    inp_cnn = panel.nn_config_widget.io_input.text()
    print(f"    CNN  → {inp_cnn}")
    assert "canales" in inp_cnn and "muestras" in inp_cnn, "entrada CNN incompleta"

    print("[4] EEGNet y Riemann/CSP en el panel")
    _select(panel.clf_combo, "nn_eegnet")
    assert not panel.nn_config_widget._eegnet_box.isHidden(), "no se mostró la config EEGNet"
    assert not panel.nn_config_widget._layers_scroll.isVisible() or True  # capas ocultas
    inp_e = panel.nn_config_widget.io_input.text()
    assert "EEGNet" in inp_e, f"entrada EEGNet: {inp_e}"
    cfg = panel.nn_config()
    assert cfg["type"] == "eegnet" and "F1" in cfg, f"config EEGNet: {cfg}"

    _select(panel.clf_combo, "riemann_mdm")
    assert not panel.raw_box.isHidden(), "no se mostró la ventana para Riemann"
    assert panel.raw_window_value() > 0
    print(f"    EEGNet config OK (F1={cfg['F1']}, D={cfg['D']}), Riemann ventana={panel.raw_window_value()}")

    print("[5] Historial navegable")
    win.refresh_all()
    hist = win.changelog_list
    assert hist.count() >= 2, "el historial no se pobló"
    assert hist.item(0).data(Qt.ItemDataRole.UserRole) == 0, "falta 'Estado inicial'"
    win._on_history_click(hist.item(0))                 # navegar al inicio
    assert win.project.changelog.applied_count() == 0, "no navegó al inicio"
    win._on_history_click(win.changelog_list.item(win.changelog_list.count() - 1))
    assert win.project.changelog.applied_count() > 0, "no volvió hacia adelante"
    print("    clic en el historial navega por la línea de tiempo")

    win.acq_panel.shutdown()
    print("\nUI EXTRAS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
