"""Paneles laterales: preprocesamiento, dataset y clasificación.

Cada panel actúa sobre el proyecto a través del ``controller`` (la ventana
principal), que centraliza el acceso al modelo y los refrescos de la vista.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
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

        layout.addWidget(QLabel("Pipeline (marca la casilla para activar/desactivar):"))
        self.steps_list = QListWidget()
        self.steps_list.currentRowChanged.connect(self._show_params)
        # La casilla de cada paso lo activa/desactiva (sin borrarlo).
        self.steps_list.itemChanged.connect(self._on_step_toggled)
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

        channels_btn = QPushButton("Seleccionar canales…")
        channels_btn.setToolTip("Activa/desactiva canales (p. ej. excluir los EOG). "
                                "Afecta a CAR, características y modelos.")
        channels_btn.clicked.connect(self.controller.select_channels)
        layout.addWidget(channels_btn)

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

    def _on_step_toggled(self, item: QListWidgetItem) -> None:
        """Marca/desmarca la casilla de un paso → activarlo/desactivarlo."""
        row = self.steps_list.row(item)
        if row < 0:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        self._pending_row = row          # conservar la selección tras refrescar
        self.controller.set_pipeline_step_enabled(row, enabled)

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
                enabled = step.get("enabled", True)
                item = QListWidgetItem(f"{i + 1}. {label}  [{desc}]")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if enabled
                                   else Qt.CheckState.Unchecked)
                if not enabled:                          # paso desactivado: atenuado/tachado
                    f = item.font(); f.setStrikeOut(True); item.setFont(f)
                    item.setForeground(QColor("#7c848d"))
                self.steps_list.addItem(item)
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

        # Mezcla los valores guardados con los por defecto del paso: así los pasos
        # creados antes (p. ej. sin 'design'/'numtaps') también muestran las opciones.
        merged = {**preprocessing.STEP_DEFAULTS.get(stype, {}), **step.get("params", {})}
        for key, value in merged.items():
            w = self._make_editor(key, value, stype)
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

    def _make_editor(self, key: str, value, stype: str | None = None) -> QWidget:
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
        else:  # str (método, tipo de diseño…)
            w = QComboBox()
            # El notch ofrece iir/fir; los demás filtros, butter/fir.
            design_opts = ["iir", "fir"] if stype == "notch" else ["butter", "fir"]
            options = {
                "type": ["linear", "constant"],
                "method": ["zscore", "minmax"],
                "design": design_opts,
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
        clear_btn = QPushButton("Vaciar todo")
        clear_btn.setToolTip("Elimina TODOS los segmentos de una vez (se puede deshacer).")
        clear_btn.clicked.connect(self.controller.clear_all_segments)
        seg_btns.addWidget(relabel_btn)
        seg_btns.addWidget(remove_btn)
        seg_btns.addWidget(clear_btn)
        layout.addLayout(seg_btns)

        # Crear segmentos a partir de los marcadores (Event Id) -> clases.
        mk_box = QGroupBox("Segmentos desde marcadores")
        mk_layout = QVBoxLayout(mk_box)
        mk_layout.addWidget(QLabel(
            "Convierte los marcadores en segmentos etiquetados (cada marcador = "
            "una clase)."))
        self.marker_all = QCheckBox("Todas las fuentes (no solo la abierta)")
        self.marker_all.setToolTip("Si está marcado, segmenta los marcadores de TODAS las "
                                   "fuentes del proyecto de una vez; si no, solo la actual.")
        mk_layout.addWidget(self.marker_all)
        mk_row = QHBoxLayout()
        mk_row.addWidget(QLabel("Desfase:"))
        self.marker_offset = QSpinBox()
        self.marker_offset.setRange(0, 1000000)
        self.marker_offset.setSingleStep(64)
        self.marker_offset.setValue(0)
        self.marker_offset.setToolTip(
            "Desfase en MUESTRAS a saltar tras el marcador antes de empezar el "
            "segmento (p. ej. para llegar al periodo de interés). segundos × fs = muestras.")
        mk_row.addWidget(self.marker_offset)
        mk_row.addWidget(QLabel("Ventana:"))
        self.marker_window = QSpinBox()
        self.marker_window.setRange(0, 1000000)
        self.marker_window.setSingleStep(64)
        self.marker_window.setValue(0)
        self.marker_window.setToolTip(
            "Duración del segmento en MUESTRAS (0 = hasta el siguiente marcador). "
            "segundos × fs = muestras (250 Hz → 2 s = 500).")
        mk_row.addWidget(self.marker_window)
        mk_btn = QPushButton("Crear")
        mk_btn.clicked.connect(lambda: self.controller.create_segments_from_markers(
            self.marker_window.value(), self.marker_offset.value(), self.marker_all.isChecked()))
        mk_row.addWidget(mk_btn)
        mk_layout.addLayout(mk_row)
        mk_help = QLabel(
            "Cada segmento empieza en (marcador + Desfase) y dura Ventana muestras.\n"
            "• Desfase (muestras): cuántas muestras saltar tras el marcador antes de "
            "empezar; sirve para llegar al periodo de interés (p. ej. saltar el cue).\n"
            "• Ventana (muestras): duración del segmento; 0 = hasta el siguiente marcador.\n"
            "Unidad = muestras. Para pasar a segundos: muestras = segundos × fs "
            "(p. ej. 250 Hz → 2 s = 500 muestras; EPOC+ 128 Hz → 2 s = 256).")
        mk_help.setWordWrap(True)
        mk_help.setStyleSheet("color: #8a929b; font-size: 11px;")
        mk_layout.addWidget(mk_help)
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
        view_btn = QPushButton("Ver características de la selección…")
        view_btn.setToolTip("Calcula y muestra las potencias por banda y características "
                            "temporales de la región seleccionada en el visor.")
        view_btn.clicked.connect(self.controller.show_features)
        feat_layout.addWidget(view_btn)
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
        self.controller.request_autosave()

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

        # Parámetros del Random Forest (visible solo para RF).
        self.rf_box = QGroupBox("Parámetros del Random Forest")
        rf_form = QFormLayout(self.rf_box)
        self.rf_estimators = QSpinBox()
        self.rf_estimators.setRange(1, 2000)
        self.rf_estimators.setValue(200)
        self.rf_estimators.setToolTip("Nº de árboles: más árboles = más estable, pero más lento.")
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(0, 200)
        self.rf_max_depth.setValue(0)
        self.rf_max_depth.setToolTip("Profundidad máxima de cada árbol (0 = sin límite).")
        self.rf_min_split = QSpinBox()
        self.rf_min_split.setRange(2, 100)
        self.rf_min_split.setValue(2)
        self.rf_min_split.setToolTip("Mínimo de muestras para dividir un nodo (mayor = menos sobreajuste).")
        self.rf_max_features = QComboBox()
        self.rf_max_features.addItem("sqrt", "sqrt")
        self.rf_max_features.addItem("log2", "log2")
        self.rf_max_features.addItem("todas", None)
        self.rf_max_features.setToolTip("Nº de características consideradas en cada división.")
        self.rf_criterion = QComboBox()
        self.rf_criterion.addItems(["gini", "entropy", "log_loss"])
        self.rf_criterion.setToolTip("Medida de calidad de las divisiones.")
        rf_form.addRow("Nº de árboles:", self.rf_estimators)
        rf_form.addRow("Profundidad máx.:", self.rf_max_depth)
        rf_form.addRow("Mín. para dividir:", self.rf_min_split)
        rf_form.addRow("Máx. características:", self.rf_max_features)
        rf_form.addRow("Criterio:", self.rf_criterion)
        self.rf_box.setVisible(False)
        layout.addWidget(self.rf_box)

        # Ventana para modelos de señal cruda no neuronales (Riemann/CSP).
        self.raw_box = QGroupBox("Señal cruda")
        raw_form = QFormLayout(self.raw_box)
        self.raw_window = QSpinBox()
        self.raw_window.setRange(16, 8192)
        self.raw_window.setSingleStep(32)
        self.raw_window.setValue(512)
        self.raw_window.setToolTip(
            "Longitud fija (en MUESTRAS) a la que se ajusta cada segmento antes de "
            "entrar al modelo. segundos × fs = muestras (250 Hz → 512 ≈ 2.05 s).")
        raw_form.addRow("Ventana (muestras):", self.raw_window)
        raw_help = QLabel(
            "Riemann/CSP y las redes necesitan que todos los segmentos tengan la "
            "MISMA longitud. Como cada segmento puede durar distinto, se recorta o "
            "rellena (centrado) a esta ventana fija de muestras. Mayor ventana = más "
            "contexto temporal pero más cómputo; debe caber en tus segmentos.")
        raw_help.setWordWrap(True)
        raw_help.setStyleSheet("color: #8a929b; font-size: 11px;")
        raw_form.addRow(raw_help)
        self.raw_box.setVisible(False)
        layout.addWidget(self.raw_box)

        # Configuración detallada de la red (visible solo para redes).
        self.nn_config_widget = NNConfigWidget()
        self.nn_config_widget.setVisible(False)
        self.nn_config_widget.window.valueChanged.connect(self.update_io_info)
        layout.addWidget(self.nn_config_widget)
        self._on_svm_kernel_changed()

        train_btn = QPushButton("Entrenar y añadir al proyecto")
        train_btn.clicked.connect(self.controller.train_model)
        layout.addWidget(train_btn)

        # Modelos entrenados del proyecto (se pueden tener varios).
        models_box = QGroupBox("Modelos entrenados")
        mlay = QVBoxLayout(models_box)
        self.models_list = QListWidget()
        self.models_list.itemDoubleClicked.connect(lambda *_: self._activate_selected())
        self.models_list.currentRowChanged.connect(self._show_selected_metrics)
        mlay.addWidget(self.models_list)
        row1 = QHBoxLayout()
        for text, slot in (("Activar", self._activate_selected),
                           ("Métricas…", self._metrics_selected),
                           ("Eliminar", self._remove_selected)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row1.addWidget(b)
        mlay.addLayout(row1)
        row2 = QHBoxLayout()
        exp = QPushButton("Exportar…")
        exp.clicked.connect(self._export_selected)
        imp = QPushButton("Importar…")
        imp.clicked.connect(self.controller.import_model)
        row2.addWidget(exp)
        row2.addWidget(imp)
        mlay.addLayout(row2)
        layout.addWidget(models_box)

        predict_btn = QPushButton("Clasificar selección actual (modelo activo)")
        predict_btn.setToolTip("Extrae la región seleccionada y predice su clase")
        predict_btn.clicked.connect(self.controller.predict_selection)
        layout.addWidget(predict_btn)

        self.result_label = QLabel("Sin modelos entrenados.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

    def _on_clf_changed(self) -> None:
        key = self.classifier_key
        is_nn = classification.is_nn(key)
        self.nn_config_widget.setVisible(is_nn)
        self.svm_box.setVisible(key == "svm")
        self.rf_box.setVisible(key == "random_forest")
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

    def rf_params(self) -> dict:
        return {
            "n_estimators": self.rf_estimators.value(),
            "max_depth": self.rf_max_depth.value(),     # 0 = sin límite
            "min_samples_split": self.rf_min_split.value(),
            "max_features": self.rf_max_features.currentData(),
            "criterion": self.rf_criterion.currentText(),
        }

    def classic_params(self) -> dict | None:
        """Parámetros del clasificador clásico seleccionado (SVM o RF)."""
        key = self.classifier_key
        if key == "svm":
            return self.svm_params()
        if key == "random_forest":
            return self.rf_params()
        return None

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
        self.refresh_models()

    # --- Registro de modelos ---------------------------------------------
    def refresh_models(self) -> None:
        self.models_list.blockSignals(True)
        self.models_list.clear()
        active = self.controller.active_model_name
        for name, result in self.controller.models.items():
            cv = f"{result.cv_mean * 100:.1f}%" if result.cv_scores.size else "—"
            star = "★ " if name == active else "   "
            item = QListWidgetItem(f"{star}{name}   ({cv})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            if name == active:
                f = item.font(); f.setBold(True); item.setFont(f)
            self.models_list.addItem(item)
        # Seleccionar el modelo activo para mostrar sus métricas.
        names = list(self.controller.models)
        if active in names:
            self.models_list.setCurrentRow(names.index(active))
        self.models_list.blockSignals(False)
        if not self.controller.models:
            self.result_label.setText("Sin modelos entrenados.")
        else:
            self._show_selected_metrics()

    def _selected_name(self) -> str | None:
        item = self.models_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _show_selected_metrics(self, *_):
        name = self._selected_name()
        if not name or name not in self.controller.models:
            return
        r = self.controller.models[name]
        label = classification.CLASSIFIER_LABELS.get(r.classifier_name, r.classifier_name)
        cv = (f"{r.score_label}: {r.cv_mean * 100:.1f}% (±{r.cv_std * 100:.1f})"
              if r.cv_scores.size else "Sin validación (pocos datos)")
        self.result_label.setText(f"{name} — {label}\n{cv}\nClases: {', '.join(r.classes)}")

    def _activate_selected(self):
        name = self._selected_name()
        if name:
            self.controller.activate_model(name)

    def _metrics_selected(self):
        name = self._selected_name()
        if name:
            self.controller.show_model_metrics(name)

    def _remove_selected(self):
        name = self._selected_name()
        if name:
            self.controller.remove_model(name)

    def _export_selected(self):
        name = self._selected_name()
        if name:
            self.controller.export_model(name)

    def set_result(self, text: str) -> None:
        self.result_label.setText(text)
