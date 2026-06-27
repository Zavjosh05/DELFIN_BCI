"""Paneles laterales: preprocesamiento, dataset y clasificación.

Cada panel actúa sobre el proyecto a través del ``controller`` (la ventana
principal), que centraliza el acceso al modelo y los refrescos de la vista.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import classification, neuralnet, preprocessing
from .nn_config import NNConfigWidget


# ====================================================================== #
# Preprocesamiento
# ====================================================================== #
class PreprocessingPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._param_widgets: dict[str, QWidget] = {}
        self._pending_row: int | None = None   # fila a seleccionar tras refrescar
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        add_box = QHBoxLayout()
        self.step_combo = QComboBox()
        for key, label in preprocessing.STEP_LABELS.items():
            self.step_combo.addItem(label, key)
        add_btn = QPushButton("Añadir paso")
        add_btn.clicked.connect(self._add_step)
        add_box.addWidget(self.step_combo, 1)
        add_box.addWidget(add_btn)
        layout.addLayout(add_box)

        layout.addWidget(QLabel("Pipeline (se aplica en orden):"))
        self.steps_list = QListWidget()
        self.steps_list.currentRowChanged.connect(self._show_params)
        # El paso seleccionado se resalta siempre (también cuando la lista pierde
        # el foco al pulsar ▲▼/Eliminar), para saber cuál se está modificando.
        self.steps_list.setStyleSheet(
            "QListWidget::item:selected,"
            "QListWidget::item:selected:!active {"
            " background: #2f6fb0; color: #ffffff; }"
        )
        layout.addWidget(self.steps_list, 1)

        btns = QHBoxLayout()
        for text, slot in (
            ("▲", lambda: self._move(-1)),
            ("▼", lambda: self._move(1)),
            ("Eliminar", self._remove),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        layout.addLayout(btns)

        self.params_box = QGroupBox("Parámetros del paso")
        self.params_form = QFormLayout(self.params_box)
        layout.addWidget(self.params_box)

        cache_btn = QPushButton("Guardar señal procesada en el proyecto (.npz)")
        cache_btn.setToolTip("Escribe el resultado en cache/ — nunca toca el CSV original")
        cache_btn.clicked.connect(self.controller.cache_processed)
        layout.addWidget(cache_btn)

    # --- Acciones ---------------------------------------------------------
    def _add_step(self) -> None:
        key = self.step_combo.currentData()
        if self.controller.project is not None:
            # El paso nuevo se añade al final: seleccionarlo.
            self._pending_row = len(self.controller.project.state["pipeline"])
        self.controller.add_pipeline_step(key)

    def _remove(self) -> None:
        row = self.steps_list.currentRow()
        if row >= 0:
            self._pending_row = row   # seleccionar el paso que ocupa ese lugar
            self.controller.remove_pipeline_step(row)

    def _move(self, delta: int) -> None:
        row = self.steps_list.currentRow()
        if row < 0:
            return
        target = row + delta
        if 0 <= target < self.steps_list.count():
            self._pending_row = target   # la selección sigue al paso movido
        self.controller.move_pipeline_step(row, delta)

    def refresh(self) -> None:
        proj = self.controller.project
        # Reubicar la selección en el paso modificado, o conservar la actual.
        target = self._pending_row if self._pending_row is not None else self.steps_list.currentRow()
        self._pending_row = None
        self.steps_list.blockSignals(True)
        self.steps_list.clear()
        if proj is not None:
            for i, step in enumerate(proj.state["pipeline"]):
                label = preprocessing.STEP_LABELS.get(step["type"], step["type"])
                p = step.get("params", {})
                desc = ", ".join(f"{k}={v}" for k, v in p.items()) if p else "sin parámetros"
                self.steps_list.addItem(QListWidgetItem(f"{i + 1}. {label}  [{desc}]"))
        n = self.steps_list.count()
        row = target if 0 <= target < n else (n - 1 if n else -1)
        self.steps_list.setCurrentRow(row)
        self.steps_list.blockSignals(False)
        self._show_params(row)

    # --- Editor de parámetros ---------------------------------------------
    def _show_params(self, row: int) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        proj = self.controller.project
        if proj is None or row < 0 or row >= len(proj.state["pipeline"]):
            self.params_box.setTitle("Parámetros del paso")
            self.params_form.addRow(QLabel("Selecciona un paso del pipeline."))
            return
        step = proj.state["pipeline"][row]
        stype = step["type"]
        label = preprocessing.STEP_LABELS.get(stype, stype)
        self.params_box.setTitle(f"Parámetros — {label}")

        # Descripción del filtro (qué hace).
        step_desc = preprocessing.STEP_DESCRIPTIONS.get(stype)
        if step_desc:
            d = QLabel(step_desc)
            d.setWordWrap(True)
            d.setStyleSheet("color: #aeb6bf; font-style: italic;")
            self.params_form.addRow(d)

        for key, value in step.get("params", {}).items():
            w = self._make_editor(key, value)
            help_text = preprocessing.PARAM_DESCRIPTIONS.get(key, "")
            w.setToolTip(help_text)
            self._param_widgets[key] = w
            self.params_form.addRow(key, w)
            # Descripción del parámetro (qué es y qué afecta al cambiarlo).
            if help_text:
                h = QLabel(help_text)
                h.setWordWrap(True)
                h.setStyleSheet("color: #8a929b; font-size: 11px;")
                self.params_form.addRow(h)

        if self._param_widgets:
            apply_btn = QPushButton("Aplicar parámetros")
            apply_btn.clicked.connect(lambda: self._apply_params(row))
            self.params_form.addRow(apply_btn)
        else:
            note = QLabel("Este paso no tiene parámetros configurables.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #9aa4ae; font-style: italic;")
            self.params_form.addRow(note)

    def _make_editor(self, key: str, value) -> QWidget:
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
        elif isinstance(value, int):
            w = QSpinBox()
            w.setRange(0, 10000)
            w.setValue(value)
        elif isinstance(value, float):
            w = QDoubleSpinBox()
            w.setRange(0.0, 1000.0)
            w.setDecimals(2)
            w.setSingleStep(0.5)
            w.setValue(value)
        else:  # str (método, tipo...)
            w = QComboBox()
            options = {
                "type": ["linear", "constant"],
                "method": ["zscore", "minmax"],
            }.get(key, [str(value)])
            w.addItems(options)
            w.setCurrentText(str(value))
        return w

    def _apply_params(self, row: int) -> None:
        params = {}
        for key, w in self._param_widgets.items():
            if isinstance(w, QCheckBox):
                params[key] = w.isChecked()
            elif isinstance(w, QSpinBox):
                params[key] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                params[key] = float(w.value())
            elif isinstance(w, QComboBox):
                params[key] = w.currentText()
        self.controller.update_pipeline_step(row, params)


# ====================================================================== #
# Dataset
# ====================================================================== #
class DatasetPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Segmentos etiquetados:"))
        self.segments_list = QListWidget()
        layout.addWidget(self.segments_list, 1)

        seg_btns = QHBoxLayout()
        relabel_btn = QPushButton("Reetiquetar")
        relabel_btn.clicked.connect(self._relabel)
        remove_btn = QPushButton("Eliminar")
        remove_btn.clicked.connect(self._remove)
        seg_btns.addWidget(relabel_btn)
        seg_btns.addWidget(remove_btn)
        layout.addLayout(seg_btns)

        # Crear segmentos a partir de los marcadores (Event Id) -> clases.
        mk_box = QGroupBox("Segmentos desde marcadores")
        mk_layout = QVBoxLayout(mk_box)
        mk_layout.addWidget(QLabel(
            "Convierte los marcadores de la fuente seleccionada en segmentos "
            "etiquetados (cada marcador = una clase)."))
        mk_row = QHBoxLayout()
        mk_row.addWidget(QLabel("Ventana:"))
        self.marker_window = QSpinBox()
        self.marker_window.setRange(0, 100000)
        self.marker_window.setSingleStep(64)
        self.marker_window.setValue(0)
        self.marker_window.setToolTip("Muestras tras cada marcador (0 = hasta el siguiente marcador).")
        mk_row.addWidget(self.marker_window, 1)
        mk_btn = QPushButton("Crear")
        mk_btn.clicked.connect(lambda: self.controller.create_segments_from_markers(self.marker_window.value()))
        mk_row.addWidget(mk_btn)
        mk_layout.addLayout(mk_row)
        layout.addWidget(mk_box)

        feat_box = QGroupBox("Características")
        feat_layout = QVBoxLayout(feat_box)
        self.bands_chk = QCheckBox("Potencias por banda (delta…gamma)")
        self.bands_chk.setChecked(True)
        self.time_chk = QCheckBox("Características temporales (RMS, Hjorth…)")
        self.time_chk.setChecked(True)
        self.bands_chk.stateChanged.connect(self._save_cfg)
        self.time_chk.stateChanged.connect(self._save_cfg)
        feat_layout.addWidget(self.bands_chk)
        feat_layout.addWidget(self.time_chk)
        layout.addWidget(feat_box)

        build_btn = QPushButton("Construir dataset (multiproceso)")
        build_btn.clicked.connect(self.controller.build_dataset)
        layout.addWidget(build_btn)

        save_btn = QPushButton("Guardar dataset (.npz)")
        save_btn.clicked.connect(self.controller.save_dataset)
        layout.addWidget(save_btn)

        self.info_label = QLabel("Sin dataset.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def _selected_segment_id(self) -> str | None:
        item = self.segments_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _relabel(self) -> None:
        seg_id = self._selected_segment_id()
        if seg_id:
            self.controller.relabel_segment(seg_id)

    def _remove(self) -> None:
        seg_id = self._selected_segment_id()
        if seg_id:
            self.controller.remove_segment(seg_id)

    def _save_cfg(self) -> None:
        if self.controller.project is None:
            return
        cfg = {"use_bands": self.bands_chk.isChecked(), "use_time": self.time_chk.isChecked()}
        self.controller.project.state["dataset"] = cfg

    def refresh(self) -> None:
        proj = self.controller.project
        self.segments_list.clear()
        if proj is None:
            return
        for seg in proj.state["segments"]:
            src = proj.get_source(seg["source_id"])
            alias = src["alias"] if src else "?"
            dur = (seg["stop"] - seg["start"])
            n_ch = "todos" if not seg.get("channels") else str(len(seg["channels"]))
            item = QListWidgetItem(
                f"[{seg['label']}] {alias}  {seg['start']}–{seg['stop']} ({dur} m, {n_ch} can.)"
            )
            item.setData(Qt.ItemDataRole.UserRole, seg["id"])
            self.segments_list.addItem(item)
        cfg = proj.state.get("dataset", {})
        self.bands_chk.setChecked(cfg.get("use_bands", True))
        self.time_chk.setChecked(cfg.get("use_time", True))

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)


# ====================================================================== #
# Clasificación
# ====================================================================== #
class ClassificationPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.clf_combo = QComboBox()
        torch_ok = neuralnet.torch_available()
        riemann_ok = classification.riemann_available()
        for key, label in classification.CLASSIFIER_LABELS.items():
            self.clf_combo.addItem(label, key)
            disabled = (classification.is_nn(key) and not torch_ok) or \
                       (classification.is_riemann(key) and not riemann_ok)
            if disabled:
                self.clf_combo.model().item(self.clf_combo.count() - 1).setEnabled(False)
        self.clf_combo.currentIndexChanged.connect(self._on_clf_changed)
        form.addRow("Modelo:", self.clf_combo)
        layout.addLayout(form)

        if not torch_ok:
            note = QLabel("⚠ PyTorch no está instalado: las redes neuronales están "
                          "deshabilitadas (pip install torch).")
            note.setWordWrap(True)
            note.setStyleSheet("color: #d6a23e;")
            layout.addWidget(note)

        # Parámetros del SVM (visible solo para SVM): varios kernels.
        self.svm_box = QGroupBox("Parámetros del SVM")
        svm_form = QFormLayout(self.svm_box)
        self.svm_kernel = QComboBox()
        for k, lbl in classification.SVM_KERNELS.items():
            self.svm_kernel.addItem(lbl, k)
        self.svm_kernel.setCurrentIndex(1)  # rbf
        self.svm_kernel.currentIndexChanged.connect(self._on_svm_kernel_changed)
        self.svm_C = QDoubleSpinBox()
        self.svm_C.setRange(0.01, 1000.0)
        self.svm_C.setDecimals(2)
        self.svm_C.setValue(1.0)
        self.svm_C.setToolTip("Regularización: mayor C = ajusta más a los datos (riesgo de sobreajuste).")
        self.svm_gamma = QComboBox()
        self.svm_gamma.addItems(["scale", "auto"])
        self.svm_gamma.setToolTip("Coeficiente del kernel (RBF/poly/sigmoide).")
        self.svm_degree = QSpinBox()
        self.svm_degree.setRange(1, 10)
        self.svm_degree.setValue(3)
        self.svm_degree.setToolTip("Grado del polinomio (solo kernel 'poly').")
        svm_form.addRow("Kernel:", self.svm_kernel)
        svm_form.addRow("C:", self.svm_C)
        svm_form.addRow("gamma:", self.svm_gamma)
        svm_form.addRow("Grado (poly):", self.svm_degree)
        self.svm_box.setVisible(False)
        layout.addWidget(self.svm_box)

        # Ventana para modelos de señal cruda no neuronales (Riemann/CSP).
        self.raw_box = QGroupBox("Señal cruda")
        raw_form = QFormLayout(self.raw_box)
        self.raw_window = QSpinBox()
        self.raw_window.setRange(16, 8192)
        self.raw_window.setSingleStep(32)
        self.raw_window.setValue(512)
        self.raw_window.setToolTip("Muestras por segmento usadas para la covarianza/CSP.")
        raw_form.addRow("Ventana (muestras):", self.raw_window)
        self.raw_box.setVisible(False)
        layout.addWidget(self.raw_box)

        # Configuración detallada de la red (visible solo para redes).
        self.nn_config_widget = NNConfigWidget()
        self.nn_config_widget.setVisible(False)
        self.nn_config_widget.window.valueChanged.connect(self.update_io_info)
        layout.addWidget(self.nn_config_widget)
        self._on_svm_kernel_changed()

        train_btn = QPushButton("Entrenar con el dataset")
        train_btn.clicked.connect(self.controller.train_model)
        layout.addWidget(train_btn)

        save_btn = QPushButton("Guardar modelo (.joblib)")
        save_btn.clicked.connect(self.controller.save_model)
        layout.addWidget(save_btn)

        load_btn = QPushButton("Cargar modelo…")
        load_btn.clicked.connect(self.controller.load_model)
        layout.addWidget(load_btn)

        predict_btn = QPushButton("Clasificar selección actual")
        predict_btn.setToolTip("Extrae la región seleccionada y predice su clase")
        predict_btn.clicked.connect(self.controller.predict_selection)
        layout.addWidget(predict_btn)

        self.result_label = QLabel("Sin modelo entrenado.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

    def _on_clf_changed(self) -> None:
        key = self.classifier_key
        is_nn = classification.is_nn(key)
        self.nn_config_widget.setVisible(is_nn)
        self.svm_box.setVisible(key == "svm")
        self.raw_box.setVisible(classification.is_riemann(key))
        if is_nn:
            self.nn_config_widget.set_net_type(classification.net_type(key))
            self.update_io_info()

    def _on_svm_kernel_changed(self) -> None:
        kernel = self.svm_kernel.currentData()
        self.svm_degree.setEnabled(kernel == "poly")
        self.svm_gamma.setEnabled(kernel in ("rbf", "poly", "sigmoid"))

    @property
    def classifier_key(self) -> str:
        return self.clf_combo.currentData()

    def nn_config(self) -> dict:
        return self.nn_config_widget.config()

    def svm_params(self) -> dict:
        return {
            "kernel": self.svm_kernel.currentData(),
            "C": float(self.svm_C.value()),
            "gamma": self.svm_gamma.currentText(),
            "degree": self.svm_degree.value(),
        }

    def raw_window_value(self) -> int:
        """Ventana (muestras) para modelos de señal cruda no neuronales."""
        return self.raw_window.value()

    def update_io_info(self) -> None:
        """Actualiza las etiquetas de capa de entrada/salida de la red."""
        key = self.classifier_key
        if not classification.is_nn(key):
            return
        n_classes = self.controller.class_count()
        out = f"Capa de salida (Linear): {n_classes or '?'} neuronas  (= nº de clases)"
        net = classification.net_type(key)
        if net == "mlp":
            n_feat = self.controller.feature_count()
            inp = f"Capa de entrada (Linear): {n_feat or '?'} neuronas  (= nº de características)"
        else:
            n_ch = self.controller.channel_count()
            T = self.nn_config_widget.window.value()
            role = {"cnn": "Conv1d", "eegnet": "EEGNet"}.get(net, "LSTM")
            if net == "lstm":
                inp = f"Capa de entrada (LSTM): {n_ch} características/paso × {T} pasos"
            else:
                inp = f"Capa de entrada ({role}): {n_ch} canales × {T} muestras"
        self.nn_config_widget.set_io_info(inp, out)

    def refresh(self) -> None:
        self.update_io_info()

    def set_result(self, text: str) -> None:
        self.result_label.setText(text)
