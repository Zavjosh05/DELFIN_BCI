"""Diálogo para **ver y editar** la configuración de un modelo entrenado.

Muestra los hiperparámetros con los que se entrenó y, si hay datos para ello,
permite modificarlos y **reentrenar** el modelo (conservando su nombre).

* Clásicos (Random Forest / SVM): editor de sus hiperparámetros.
* Redes neuronales: editor de los hiperparámetros escalares (épocas, batch,
  learning rate, ventana…); la arquitectura por capas se muestra de solo lectura.
* Riemann / CSP: ventana (muestras) de la señal cruda.
* LDA: no tiene hiperparámetros configurables.
"""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
)

from ..core import classification

_OPTIMIZERS = ["adam", "sgd", "rmsprop"]
_ACTIVATIONS = ["relu", "tanh", "sigmoid", "elu", "leaky_relu"]


def _rf_editor(params: dict):
    """Widgets del Random Forest, prellenados. Devuelve (groupbox, getter)."""
    params = params or {}
    box = QGroupBox("Parámetros del Random Forest")
    form = QFormLayout(box)
    n_est = QSpinBox(); n_est.setRange(1, 2000); n_est.setValue(int(params.get("n_estimators", 200)))
    depth = QSpinBox(); depth.setRange(0, 200); depth.setValue(int(params.get("max_depth", 0) or 0))
    depth.setToolTip("0 = sin límite")
    split = QSpinBox(); split.setRange(2, 100); split.setValue(int(params.get("min_samples_split", 2)))
    leaf = QSpinBox(); leaf.setRange(1, 100); leaf.setValue(int(params.get("min_samples_leaf", 1)))
    leaf.setToolTip("Mínimo de muestras por hoja (mayor = menos sobreajuste).")
    feats = QComboBox()
    feats.addItem("sqrt", "sqrt"); feats.addItem("log2", "log2"); feats.addItem("todas", None)
    cur = params.get("max_features", "sqrt")
    feats.setCurrentIndex({"sqrt": 0, "log2": 1, None: 2}.get(cur, 0))
    crit = QComboBox(); crit.addItems(["gini", "entropy", "log_loss"])
    crit.setCurrentText(params.get("criterion", "gini"))
    cw = QComboBox(); cw.addItem("ninguno", "none"); cw.addItem("balanced", "balanced")
    cw.setCurrentIndex(1 if params.get("class_weight") == "balanced" else 0)
    cw.setToolTip("«balanced» compensa clases desbalanceadas.")
    form.addRow("Nº de árboles:", n_est)
    form.addRow("Profundidad máx.:", depth)
    form.addRow("Mín. para dividir:", split)
    form.addRow("Mín. por hoja:", leaf)
    form.addRow("Máx. características:", feats)
    form.addRow("Criterio:", crit)
    form.addRow("Peso de clases:", cw)

    def getter():
        return {
            "n_estimators": n_est.value(),
            "max_depth": depth.value(),
            "min_samples_split": split.value(),
            "min_samples_leaf": leaf.value(),
            "max_features": feats.currentData(),
            "criterion": crit.currentText(),
            "class_weight": cw.currentData(),
        }
    return box, getter


def _svm_editor(params: dict):
    params = params or {}
    box = QGroupBox("Parámetros del SVM")
    form = QFormLayout(box)
    kernel = QComboBox()
    for k, lbl in classification.SVM_KERNELS.items():
        kernel.addItem(lbl, k)
    idx = list(classification.SVM_KERNELS).index(params.get("kernel", "rbf")) \
        if params.get("kernel", "rbf") in classification.SVM_KERNELS else 1
    kernel.setCurrentIndex(idx)
    c = QDoubleSpinBox(); c.setRange(0.01, 1000.0); c.setDecimals(2); c.setValue(float(params.get("C", 1.0)))
    gamma = QComboBox(); gamma.addItems(["scale", "auto"]); gamma.setCurrentText(params.get("gamma", "scale"))
    degree = QSpinBox(); degree.setRange(1, 10); degree.setValue(int(params.get("degree", 3)))
    coef0 = QDoubleSpinBox(); coef0.setRange(-100.0, 100.0); coef0.setDecimals(2)
    coef0.setValue(float(params.get("coef0", 0.0)))
    coef0.setToolTip("Término independiente del kernel (solo poly/sigmoide).")
    cw = QComboBox(); cw.addItem("ninguno", "none"); cw.addItem("balanced", "balanced")
    cw.setCurrentIndex(1 if params.get("class_weight") == "balanced" else 0)
    cw.setToolTip("«balanced» compensa clases desbalanceadas.")
    form.addRow("Kernel:", kernel)
    form.addRow("C:", c)
    form.addRow("gamma:", gamma)
    form.addRow("Grado (poly):", degree)
    form.addRow("coef0 (poly/sigmoide):", coef0)
    form.addRow("Peso de clases:", cw)

    def _sync():
        k = kernel.currentData()
        degree.setEnabled(k == "poly")
        gamma.setEnabled(k in ("rbf", "poly", "sigmoid"))
        coef0.setEnabled(k in ("poly", "sigmoid"))
    kernel.currentIndexChanged.connect(_sync)
    _sync()

    def getter():
        return {
            "kernel": kernel.currentData(),
            "C": float(c.value()),
            "gamma": gamma.currentText(),
            "degree": degree.value(),
            "coef0": float(coef0.value()),
            "class_weight": cw.currentData(),
        }
    return box, getter


def _nn_editor(config: dict):
    """Editor de los escalares de la red; capas/estructura de solo lectura."""
    config = dict(config or {})
    box = QGroupBox("Configuración de la red")
    form = QFormLayout(box)
    getters: dict[str, callable] = {}
    for key, val in config.items():
        if key == "type" or isinstance(val, (list, dict)):
            ro = QLabel(json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else str(val))
            ro.setWordWrap(True)
            ro.setStyleSheet("color: #8a929b;")
            form.addRow(f"{key} (fijo):", ro)
            continue
        if isinstance(val, bool):
            w = QCheckBox(); w.setChecked(val)
            getters[key] = (lambda w=w: w.isChecked())
        elif isinstance(val, int):
            w = QSpinBox(); w.setRange(1, 100000); w.setValue(val)
            getters[key] = (lambda w=w: w.value())
        elif isinstance(val, float):
            w = QDoubleSpinBox(); w.setDecimals(5); w.setRange(0.0, 1e6); w.setValue(val)
            getters[key] = (lambda w=w: float(w.value()))
        elif key == "optimizer":
            w = QComboBox(); w.addItems(_OPTIMIZERS)
            if val in _OPTIMIZERS:
                w.setCurrentText(val)
            getters[key] = (lambda w=w: w.currentText())
        else:
            w = QLineEdit(str(val))
            getters[key] = (lambda w=w: w.text())
        form.addRow(f"{key}:", w)

    def getter():
        new = dict(config)
        for k, g in getters.items():
            new[k] = g()
        return new
    return box, getter


def _lda_editor(params: dict):
    """Widgets del LDA (solver + shrinkage). Devuelve (groupbox, getter)."""
    params = params or {}
    box = QGroupBox("Parámetros del LDA")
    form = QFormLayout(box)
    solver = QComboBox(); solver.addItems(["svd", "lsqr", "eigen"])
    solver.setCurrentText(params.get("solver", "svd"))
    shrink = QComboBox()
    shrink.addItem("ninguno", "none"); shrink.addItem("auto (Ledoit-Wolf)", "auto")
    shrink.setCurrentIndex(1 if params.get("shrinkage") == "auto" else 0)
    shrink.setToolTip("Regularización (solo lsqr/eigen); útil con pocas muestras (EEG).")
    form.addRow("Solver:", solver)
    form.addRow("Shrinkage:", shrink)

    def _sync():
        shrink.setEnabled(solver.currentText() != "svd")
    solver.currentIndexChanged.connect(_sync)
    _sync()

    def getter():
        return {"solver": solver.currentText(), "shrinkage": shrink.currentData()}
    return box, getter


def _riemann_editor(result):
    box = QGroupBox("Señal cruda (Riemann/CSP)")
    form = QFormLayout(box)
    win = QSpinBox(); win.setRange(16, 8192); win.setSingleStep(32)
    win.setValue(int(getattr(result, "raw_window", 0) or 512))
    win.setToolTip("Longitud fija (muestras) a la que se ajusta cada segmento.")
    form.addRow("Ventana (muestras):", win)

    def getter():
        return win.value()
    return box, getter


def describe_model_config(entry: dict) -> str:
    """Resumen legible de los hiperparámetros de una configuración de modelo."""
    key = entry.get("classifier_name", "")
    if classification.is_nn(key):
        cfg = entry.get("nn_config") or {}
        bits = [f"{k}={cfg[k]}" for k in
                ("type", "epochs", "batch_size", "learning_rate", "window_samples")
                if k in cfg]
        return ", ".join(bits) or "configuración de red"
    if classification.is_riemann(key):
        return f"ventana={entry.get('raw_window') or '—'} muestras"
    params = entry.get("clf_params") or {}
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "sin hiperparámetros"


def choose_imported_configs(parent, entries: list, can_features: bool, can_raw: bool):
    """Ofrece **reutilizar** los hiperparámetros de una config/bundle importada
    entrenando con los datos del proyecto actual.

    Marca como no disponibles las que necesitan datos que aún no existen (dataset
    de características o segmentos etiquetados). Devuelve la lista de entradas
    elegidas, o ``None`` si se cancela.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Configuraciones de modelo importadas")
    dlg.resize(580, 400)
    lay = QVBoxLayout(dlg)
    info = QLabel(
        "El archivo importado trae <b>configuraciones de clasificador</b> "
        "(hiperparámetros). Puedes entrenarlas con los datos de <b>este</b> proyecto; "
        "se añaden como modelos nuevos y no sustituyen a los modelos importados.")
    info.setWordWrap(True)
    lay.addWidget(info)

    lst = QListWidget()
    lst.setWordWrap(True)                            # evita el desplazamiento horizontal
    for e in entries:
        key = e.get("classifier_name", "")
        label = classification.CLASSIFIER_LABELS.get(key, key)
        needs_raw = classification.requires_raw(key)
        available = can_raw if needs_raw else can_features
        text = f"{e.get('name', key)}  —  {label}\n{describe_model_config(e)}"
        if not available:
            text += ("\n⚠ Necesita segmentos etiquetados en este proyecto."
                     if needs_raw else "\n⚠ Necesita un dataset construido en este proyecto.")
        it = QListWidgetItem(text)
        it.setData(Qt.ItemDataRole.UserRole, e)
        if available:
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
        else:
            it.setFlags(Qt.ItemFlag.NoItemFlags)     # no seleccionable: faltan datos
        lst.addItem(it)
    lay.addWidget(lst)

    bb = QDialogButtonBox()
    train_btn = bb.addButton("Entrenar con mis datos",
                             QDialogButtonBox.ButtonRole.AcceptRole)
    bb.addButton("Ahora no", QDialogButtonBox.ButtonRole.RejectRole)
    train_btn.clicked.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return [lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lst.count())
            if lst.item(i).checkState() == Qt.CheckState.Checked]


def edit_model_config(parent, name: str, result, retrainable: bool, reason: str = ""):
    """Muestra la configuración del modelo y, si procede, permite reentrenar.

    Devuelve ``(kind, payload)`` para reentrenar — ``kind`` ∈
    {"classic", "nn", "riemann"} — o ``None`` si se cierra sin reentrenar.
    """
    fam = classification.MODEL_FAMILY.get(result.classifier_name, "classic")
    label = classification.CLASSIFIER_LABELS.get(result.classifier_name, result.classifier_name)

    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Configuración — {name}")
    dlg.resize(460, 360)
    lay = QVBoxLayout(dlg)

    hdr = QLabel(f"<b>{name}</b> — {label}<br>"
                 f"Clases: {', '.join(map(str, result.classes))}<br>"
                 f"Entrada: {'señal cruda' if result.input_kind == 'raw' else 'características'}"
                 f"  ·  muestras: {getattr(result, 'n_samples', 0)}")
    hdr.setWordWrap(True)
    lay.addWidget(hdr)

    getter = None
    kind = None
    if fam == "classic":
        if result.classifier_name == "svm":
            box, getter = _svm_editor(result.clf_params)
        elif result.classifier_name == "lda":
            box, getter = _lda_editor(result.clf_params)
        else:
            box, getter = _rf_editor(result.clf_params)
        lay.addWidget(box)
        kind = "classic"
    elif fam == "nn":
        box, getter = _nn_editor(result.nn_config or {})
        lay.addWidget(box)
        kind = "nn"
    elif fam == "riemann":
        box, getter = _riemann_editor(result)
        lay.addWidget(box)
        kind = "riemann"

    if not retrainable and reason:
        warn = QLabel(f"⚠ Para reentrenar: {reason}")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #d6a23e;")
        lay.addWidget(warn)

    bb = QDialogButtonBox()
    retrain_btn = None
    if getter is not None:
        retrain_btn = bb.addButton("Reentrenar", QDialogButtonBox.ButtonRole.AcceptRole)
        retrain_btn.setEnabled(retrainable)
        retrain_btn.setToolTip("Aplica los cambios y vuelve a entrenar el modelo "
                               "(sustituye al actual con el mismo nombre).")
    bb.addButton(QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.reject)
    if retrain_btn is not None:
        retrain_btn.clicked.connect(dlg.accept)
    lay.addWidget(bb)

    if dlg.exec() != QDialog.DialogCode.Accepted or getter is None:
        return None
    return kind, getter()
