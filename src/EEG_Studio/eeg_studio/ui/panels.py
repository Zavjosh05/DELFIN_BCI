"""Paneles laterales: preprocesamiento, dataset y clasificación.

Cada panel actúa sobre el proyecto a través del ``controller`` (la ventana
principal), que centraliza el acceso al modelo y los refrescos de la vista.
"""
from __future__ import annotations

import copy

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from ..core import augment, classification, neuralnet, preprocessing
from .nn_config import NNConfigWidget

# Entrada siempre presente en la lista de configuraciones: los valores por
# defecto del programa (no se guardan en el proyecto ni se pueden borrar).
DEFAULT_CONFIG_NAME = "· Valores por defecto ·"


# ====================================================================== #
# Preprocesamiento
# ====================================================================== #
class PreprocessingPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._param_widgets: dict[str, QWidget] = {}
        self._pending_row: int | None = None   # fila a seleccionar tras refrescar
        self._syncing_pipelines = False        # evita recursión al repoblar la barra
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Varios pipelines por proyecto, como pestañas de navegador.
        pl_row = QHBoxLayout()
        pl_row.addWidget(QLabel("Pipelines:"))
        self.pipeline_bar = QTabBar()
        self.pipeline_bar.setExpanding(False)
        self.pipeline_bar.setDrawBase(False)
        self.pipeline_bar.setTabsClosable(True)
        self.pipeline_bar.setUsesScrollButtons(True)
        self.pipeline_bar.setElideMode(Qt.TextElideMode.ElideRight)
        self.pipeline_bar.setToolTip("Cada pestaña es un pipeline de preprocesamiento "
                                     "independiente. Doble clic para renombrar.")
        self.pipeline_bar.currentChanged.connect(self._on_pipeline_changed)
        self.pipeline_bar.tabCloseRequested.connect(self._on_pipeline_close)
        self.pipeline_bar.tabBarDoubleClicked.connect(self._on_pipeline_rename)
        add_pl_btn = QPushButton("＋")
        add_pl_btn.setFixedWidth(30)
        add_pl_btn.setToolTip("Añadir un pipeline nuevo.")
        add_pl_btn.clicked.connect(lambda: self.controller.add_pipeline())
        del_pl_btn = QPushButton("🗑")
        del_pl_btn.setFixedWidth(30)
        del_pl_btn.setToolTip("Eliminar el pipeline activo (reversible con Ctrl+Z).")
        del_pl_btn.clicked.connect(self._delete_current_pipeline)
        pl_row.addWidget(self.pipeline_bar, 1)
        pl_row.addWidget(add_pl_btn)
        pl_row.addWidget(del_pl_btn)
        layout.addLayout(pl_row)

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

        cache_btn = QPushButton("Guardar señal procesada (.npz)")
        cache_btn.setToolTip("Guarda la señal procesada en el proyecto (cache/) — "
                             "nunca toca el CSV original.")
        cache_btn.clicked.connect(self.controller.cache_processed)
        layout.addWidget(cache_btn)

    # --- Pipelines (pestañas) ---------------------------------------------
    def _on_pipeline_changed(self, index: int) -> None:
        if self._syncing_pipelines or index < 0:
            return
        self.controller.set_active_pipeline(index)

    def _on_pipeline_close(self, index: int) -> None:
        if not self._syncing_pipelines:
            self.controller.remove_pipeline(index)

    def _delete_current_pipeline(self) -> None:
        proj = self.controller.project
        if proj is None:
            return
        i = proj.active_pipeline_index()
        pls = proj.pipelines()
        if len(pls) <= 1:
            self.controller.remove_pipeline(i)      # el controlador avisa de que debe quedar 1
            return
        name = pls[i]["name"]
        if QMessageBox.question(
                self, "Eliminar pipeline",
                f"¿Eliminar el pipeline «{name}»?\n(reversible con Ctrl+Z)"
        ) == QMessageBox.StandardButton.Yes:
            self.controller.remove_pipeline(i)

    def _on_pipeline_rename(self, index: int) -> None:
        proj = self.controller.project
        if proj is None or not (0 <= index < len(proj.pipelines())):
            return
        current = proj.pipelines()[index]["name"]
        name, ok = QInputDialog.getText(self, "Renombrar pipeline", "Nombre:", text=current)
        if ok and name.strip():
            self.controller.rename_pipeline(index, name.strip())

    def _refresh_pipeline_bar(self, proj) -> None:
        self._syncing_pipelines = True
        bar = self.pipeline_bar
        while bar.count():
            bar.removeTab(0)
        if proj is not None:
            pls = proj.pipelines()
            for pl in pls:
                bar.addTab(pl["name"])
            bar.setTabsClosable(len(pls) > 1)   # nunca cerrar el último
            bar.setCurrentIndex(proj.active_pipeline_index())
        self._syncing_pipelines = False

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
        self._refresh_pipeline_bar(proj)
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

        # El paso ICA ofrece un análisis espacial: mapas topográficos de los
        # componentes para VER dónde surgen los artefactos que se eliminan.
        if stype == "ica":
            ica_btn = QPushButton("Ver mapas espaciales (ICA)…")
            ica_btn.setToolTip(
                "Descompone la fuente abierta con ICA (aplicando antes los pasos "
                "previos del pipeline) y muestra el mapa topográfico de cada "
                "componente sobre la cabeza, resaltando los candidatos a artefacto."
            )
            ica_btn.clicked.connect(lambda: self.controller.show_ica_topomaps(row))
            self.params_form.addRow(ica_btn)
            hint = QLabel("Rojo/azul = peso del componente por zona del cuero "
                          "cabelludo; ⚠ = kurtosis alta (artefacto).")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #8a929b; font-size: 11px;")
            self.params_form.addRow(hint)

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
                "mode": ["manual", "automatico"],
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

        # Resumen: total de muestras y nº por clase.
        self.class_summary = QLabel("Sin segmentos.")
        self.class_summary.setWordWrap(True)
        self.class_summary.setStyleSheet("color: #9be7c4; font-weight: 600;")
        layout.addWidget(self.class_summary)

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
        view_btn = QPushButton("Ver características…")
        view_btn.setToolTip("Calcula y muestra las potencias por banda y características "
                            "temporales de la región seleccionada en el visor.")
        view_btn.clicked.connect(self.controller.show_features)
        feat_layout.addWidget(view_btn)
        layout.addWidget(feat_box)

        build_btn = QPushButton("Construir dataset")
        build_btn.setToolTip("Construye el dataset (extracción de características en multiproceso).")
        build_btn.clicked.connect(self.controller.build_dataset)
        layout.addWidget(build_btn)

        save_btn = QPushButton("Guardar dataset (.npz)")
        save_btn.clicked.connect(self.controller.save_dataset)
        layout.addWidget(save_btn)

        import_btn = QPushButton("Importar dataset (.npz)…")
        import_btn.setToolTip(
            "Carga un dataset ya construido en otra sesión (.npz) y lo deja listo "
            "para entrenar, sin volver a extraer características ni necesitar los "
            "CSV originales.")
        import_btn.clicked.connect(self.controller.import_dataset)
        layout.addWidget(import_btn)

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

        # Total de muestras (segmentos) y desglose por clase.
        segs = proj.state["segments"]
        if segs:
            from collections import Counter
            counts = Counter(s["label"] for s in segs)
            por_clase = " · ".join(f"{lab}: {n}" for lab, n in sorted(counts.items()))
            self.class_summary.setText(
                f"Total: {len(segs)} muestras · {len(counts)} clases  ⟶  {por_clase}")
        else:
            self.class_summary.setText("Sin segmentos.")

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
        self.svm_coef0 = QDoubleSpinBox()
        self.svm_coef0.setRange(-100.0, 100.0)
        self.svm_coef0.setDecimals(2)
        self.svm_coef0.setValue(0.0)
        self.svm_coef0.setToolTip("Término independiente del kernel (solo 'poly' y 'sigmoide').")
        self.svm_class_weight = QComboBox()
        self.svm_class_weight.addItem("ninguno", "none")
        self.svm_class_weight.addItem("balanced", "balanced")
        self.svm_class_weight.setToolTip("«balanced» compensa clases desbalanceadas.")
        svm_form.addRow("Kernel:", self.svm_kernel)
        svm_form.addRow("C:", self.svm_C)
        svm_form.addRow("gamma:", self.svm_gamma)
        svm_form.addRow("Grado (poly):", self.svm_degree)
        svm_form.addRow("coef0 (poly/sigmoide):", self.svm_coef0)
        svm_form.addRow("Peso de clases:", self.svm_class_weight)
        self.svm_box.setVisible(False)
        layout.addWidget(self.svm_box)

        # Parámetros del LDA (visible solo para LDA): solver + shrinkage.
        self.lda_box = QGroupBox("Parámetros del LDA")
        lda_form = QFormLayout(self.lda_box)
        self.lda_solver = QComboBox()
        self.lda_solver.addItems(["svd", "lsqr", "eigen"])
        self.lda_solver.setToolTip("svd: sin regularización (por defecto). lsqr/eigen: "
                                   "permiten «shrinkage».")
        self.lda_solver.currentIndexChanged.connect(self._on_lda_solver_changed)
        self.lda_shrinkage = QComboBox()
        self.lda_shrinkage.addItem("ninguno", "none")
        self.lda_shrinkage.addItem("auto (Ledoit-Wolf)", "auto")
        self.lda_shrinkage.setToolTip("Regularización de la covarianza. «auto» ayuda con "
                                      "pocas muestras y muchas características (típico en EEG). "
                                      "Requiere solver lsqr/eigen.")
        lda_form.addRow("Solver:", self.lda_solver)
        lda_form.addRow("Shrinkage:", self.lda_shrinkage)
        self.lda_box.setVisible(False)
        layout.addWidget(self.lda_box)

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
        self.rf_min_leaf = QSpinBox()
        self.rf_min_leaf.setRange(1, 100)
        self.rf_min_leaf.setValue(1)
        self.rf_min_leaf.setToolTip("Mínimo de muestras por hoja (mayor = árboles más suaves, "
                                    "menos sobreajuste; útil con datasets pequeños).")
        self.rf_max_features = QComboBox()
        self.rf_max_features.addItem("sqrt", "sqrt")
        self.rf_max_features.addItem("log2", "log2")
        self.rf_max_features.addItem("todas", None)
        self.rf_max_features.setToolTip("Nº de características consideradas en cada división.")
        self.rf_criterion = QComboBox()
        self.rf_criterion.addItems(["gini", "entropy", "log_loss"])
        self.rf_criterion.setToolTip("Medida de calidad de las divisiones.")
        self.rf_class_weight = QComboBox()
        self.rf_class_weight.addItem("ninguno", "none")
        self.rf_class_weight.addItem("balanced", "balanced")
        self.rf_class_weight.setToolTip("«balanced» compensa clases desbalanceadas "
                                        "(distinto nº de ensayos por clase).")
        rf_form.addRow("Nº de árboles:", self.rf_estimators)
        rf_form.addRow("Profundidad máx.:", self.rf_max_depth)
        rf_form.addRow("Mín. para dividir:", self.rf_min_split)
        rf_form.addRow("Mín. por hoja:", self.rf_min_leaf)
        rf_form.addRow("Máx. características:", self.rf_max_features)
        rf_form.addRow("Criterio:", self.rf_criterion)
        rf_form.addRow("Peso de clases:", self.rf_class_weight)
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

        # Estrategia multiclase: descomponer N clases en clasificadores binarios.
        # Aplica a clásicos y a Riemann/CSP (las redes ya salen con N neuronas).
        self.multiclass_box = QGroupBox("Estrategia multiclase")
        mc_form = QFormLayout(self.multiclass_box)
        self.multiclass_combo = QComboBox()
        for key, label in classification.MULTICLASS_STRATEGIES.items():
            self.multiclass_combo.addItem(label, key)
        self.multiclass_combo.currentIndexChanged.connect(self._on_multiclass_changed)
        mc_form.addRow("Estrategia:", self.multiclass_combo)
        self.multiclass_help = QLabel()
        self.multiclass_help.setWordWrap(True)
        self.multiclass_help.setStyleSheet("color: #8a929b; font-size: 11px;")
        mc_form.addRow(self.multiclass_help)
        self.multiclass_box.setVisible(False)
        layout.addWidget(self.multiclass_box)

        # Configuración detallada de la red (visible solo para redes).
        self.nn_config_widget = NNConfigWidget()
        self.nn_config_widget.setVisible(False)
        self.nn_config_widget.window.valueChanged.connect(self.update_io_info)
        layout.addWidget(self.nn_config_widget)
        self._on_svm_kernel_changed()
        self._on_lda_solver_changed()

        # --- Aumento de datos (data augmentation) -----------------------------
        self.aug_box = QGroupBox("Aumento de datos (solo al entrenar)")
        self.aug_box.setCheckable(True)
        self.aug_box.setChecked(False)
        self.aug_box.setToolTip(
            "Genera copias perturbadas de los ensayos para que el modelo aprenda el "
            "patrón y no el ensayo concreto. Se aplica SOLO a los datos de "
            "entrenamiento: la validación se mide con los ensayos reales.")
        aug_lay = QVBoxLayout(self.aug_box)
        aug_form = QFormLayout()
        self.aug_copies = QSpinBox()
        self.aug_copies.setRange(1, 20)
        self.aug_copies.setValue(1)
        self.aug_copies.setToolTip("Copias aumentadas por cada ensayo original "
                                   "(1 = duplica el conjunto de entrenamiento).")
        self.aug_prob = QDoubleSpinBox()
        self.aug_prob.setRange(0.0, 1.0)
        self.aug_prob.setSingleStep(0.1)
        self.aug_prob.setDecimals(2)
        self.aug_prob.setValue(0.5)
        self.aug_prob.setToolTip(
            "Probabilidad de aplicar CADA técnica activa a cada copia: así cada copia "
            "es una combinación distinta (aumentación automática).")
        aug_form.addRow("Copias por ensayo:", self.aug_copies)
        aug_form.addRow("Prob. por técnica:", self.aug_prob)
        aug_lay.addLayout(aug_form)

        # Una fila por técnica: casilla + su nivel.
        self._aug_checks: dict[str, QCheckBox] = {}
        self._aug_levels: dict[str, QDoubleSpinBox] = {}
        for key, label in augment.TECHNIQUES.items():
            row = QHBoxLayout()
            chk = QCheckBox(label)
            chk.setChecked(key in ("noise", "amplitude"))
            chk.setToolTip(augment.TECHNIQUE_DESCRIPTIONS.get(key, ""))
            lvl = QDoubleSpinBox()
            lvl.setRange(0.0, 1.0)
            lvl.setSingleStep(0.05)
            lvl.setDecimals(2)
            lvl.setValue(augment.DEFAULT_LEVELS.get(key, 0.1))
            lvl.setToolTip(augment.TECHNIQUE_DESCRIPTIONS.get(key, ""))
            row.addWidget(chk, 1)
            row.addWidget(QLabel("nivel:"))
            row.addWidget(lvl)
            aug_lay.addLayout(row)
            self._aug_checks[key] = chk
            self._aug_levels[key] = lvl
        aug_help = QLabel(
            "Útil sobre todo con las REDES (que sobreajustan con pocos ensayos); a LDA "
            "con shrinkage o a Riemann les aporta poco. La «Traslación circular» solo "
            "aplica a modelos de señal cruda.")
        aug_help.setWordWrap(True)
        aug_help.setStyleSheet("color: #8a929b; font-size: 11px;")
        aug_lay.addWidget(aug_help)
        layout.addWidget(self.aug_box)

        # Configuraciones de modelo: hiperparámetros con nombre, guardables en el
        # proyecto SIN entrenar nada (recetas reutilizables).
        self.cfg_box = QGroupBox("Configuraciones de modelo (sin entrenar)")
        cfg_lay = QVBoxLayout(self.cfg_box)
        self.cfg_combo = QComboBox()
        self.cfg_combo.setToolTip(
            "Configuraciones guardadas para este clasificador (solo hiperparámetros). "
            "Cárgalas en los campos y entrena cuando quieras.")
        cfg_lay.addWidget(self.cfg_combo)
        cfg_row = QHBoxLayout()
        for text, slot, tip in (
                ("Cargar", self.load_selected_config,
                 "Vuelca la configuración elegida en los campos. No entrena."),
                ("Guardar actual…", self.save_current_config,
                 "Guarda los valores actuales en el proyecto con un nombre. No entrena."),
                ("Eliminar", self.remove_selected_config,
                 "Quita la configuración guardada del proyecto.")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            cfg_row.addWidget(b)
        cfg_lay.addLayout(cfg_row)
        self.train_all_btn = QPushButton("Entrenar TODAS las configuraciones guardadas")
        self.train_all_btn.setToolTip(
            "Entrena, una tras otra, un modelo por cada configuración guardada en el "
            "proyecto (de cualquier clasificador). Las que necesiten datos que no "
            "tienes se omiten.")
        self.train_all_btn.clicked.connect(
            lambda: self.controller.train_all_saved_configs())
        cfg_lay.addWidget(self.train_all_btn)
        layout.addWidget(self.cfg_box)

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
                           ("Configuración…", self._configure_selected),
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
        self.retrain_all_btn = QPushButton("Reentrenar TODOS con los datos actuales")
        self.retrain_all_btn.setToolTip(
            "Vuelve a entrenar cada modelo ya entrenado con sus MISMOS "
            "hiperparámetros pero con los datos actuales del proyecto (útil si "
            "cambiaste el dataset o los segmentos). Sustituye cada modelo por su "
            "versión nueva, conservando el nombre.")
        self.retrain_all_btn.clicked.connect(lambda: self.controller.retrain_all_models())
        mlay.addWidget(self.retrain_all_btn)
        layout.addWidget(models_box)

        predict_btn = QPushButton("Clasificar selección")
        predict_btn.setToolTip("Extrae la región seleccionada en el visor y predice su "
                               "clase con el modelo activo.")
        predict_btn.clicked.connect(self.controller.predict_selection)
        layout.addWidget(predict_btn)

        self.result_label = QLabel("Sin modelos entrenados.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

        self._snapshot_defaults()

    def _snapshot_defaults(self) -> None:
        """Guarda los valores POR DEFECTO de cada clasificador, leyéndolos de los
        propios campos recién construidos (no se duplican en ningún sitio).

        Sirven para ofrecer «Valores por defecto» y poder volver atrás."""
        aug = augment.default_config()          # apagado, como recién abierto
        self._default_configs: dict[str, dict] = {
            "random_forest": {"classifier_name": "random_forest",
                              "augment_config": aug,
                              "clf_params": self.rf_params()},
            "svm": {"classifier_name": "svm", "clf_params": self.svm_params()},
            "lda": {"classifier_name": "lda", "clf_params": self.lda_params()},
        }
        for key in classification.CLASSIFIER_LABELS:
            if classification.is_nn(key):
                self._default_configs[key] = {
                    "classifier_name": key, "augment_config": aug,
                    "nn_config": neuralnet.default_config(classification.net_type(key))}
            elif classification.is_riemann(key):
                self._default_configs[key] = {"classifier_name": key,
                                              "augment_config": aug,
                                              "raw_window": self.raw_window.value(),
                                              "clf_params": self.riemann_params()}
            else:
                self._default_configs.setdefault(key, {}).setdefault(
                    "augment_config", aug)

    def default_config_dict(self, classifier_name: str | None = None) -> dict | None:
        """Configuración por defecto del clasificador (la de fábrica del programa)."""
        key = classifier_name or self.classifier_key
        cfg = self._default_configs.get(key)
        return copy.deepcopy(cfg) if cfg else None

    def _on_multiclass_changed(self) -> None:
        """Muestra qué hace la estrategia elegida (y su coste en nº de modelos)."""
        key = self.multiclass_combo.currentData() or "nativa"
        text = classification.MULTICLASS_DESCRIPTIONS.get(key, "")
        n = self.controller.class_count()
        if n and n > 2:
            if key == "ovo":
                text += f"  → con tus {n} clases: {n * (n - 1) // 2} modelos binarios."
            elif key == "ovr":
                text += f"  → con tus {n} clases: {n} modelos binarios."
        self.multiclass_help.setText(text)

    def _on_clf_changed(self) -> None:
        key = self.classifier_key
        is_nn = classification.is_nn(key)
        self.nn_config_widget.setVisible(is_nn)
        self.svm_box.setVisible(key == "svm")
        self.rf_box.setVisible(key == "random_forest")
        self.lda_box.setVisible(key == "lda")
        self.raw_box.setVisible(classification.is_riemann(key))
        # Las redes no usan descomposición binaria (su capa de salida ya es N clases).
        self.multiclass_box.setVisible(not is_nn)
        self._on_multiclass_changed()
        if is_nn:
            self.nn_config_widget.set_net_type(classification.net_type(key))
            self.update_io_info()
        self.refresh_model_configs()

    def _on_svm_kernel_changed(self) -> None:
        kernel = self.svm_kernel.currentData()
        self.svm_degree.setEnabled(kernel == "poly")
        self.svm_gamma.setEnabled(kernel in ("rbf", "poly", "sigmoid"))
        self.svm_coef0.setEnabled(kernel in ("poly", "sigmoid"))

    def _on_lda_solver_changed(self) -> None:
        # El shrinkage solo aplica a lsqr/eigen (no a svd).
        self.lda_shrinkage.setEnabled(self.lda_solver.currentText() != "svd")

    # --- Configuraciones de modelo (hiperparámetros, sin entrenar) --------
    # --- Aumento de datos --------------------------------------------------
    def augment_config(self) -> dict:
        """Configuración de aumento según los campos (siempre un dict completo)."""
        cfg = augment.default_config()
        cfg["enabled"] = self.aug_box.isChecked()
        cfg["copies"] = self.aug_copies.value()
        cfg["probability"] = float(self.aug_prob.value())
        cfg["techniques"] = {k: c.isChecked() for k, c in self._aug_checks.items()}
        cfg["levels"] = {k: float(s.value()) for k, s in self._aug_levels.items()}
        return cfg

    def set_augment_config(self, cfg: dict | None) -> None:
        """Vuelca una configuración de aumento en los campos."""
        cfg = cfg or augment.default_config()
        self.aug_box.setChecked(bool(cfg.get("enabled")))
        self.aug_copies.setValue(int(cfg.get("copies", 1)))
        self.aug_prob.setValue(float(cfg.get("probability", 0.5)))
        for k, chk in self._aug_checks.items():
            chk.setChecked(bool((cfg.get("techniques") or {}).get(k, False)))
        for k, spin in self._aug_levels.items():
            spin.setValue(float((cfg.get("levels") or {}).get(
                k, augment.DEFAULT_LEVELS.get(k, 0.1))))

    def current_config_dict(self) -> dict:
        """Los valores actuales de los campos como configuración de modelo."""
        key = self.classifier_key
        cfg: dict = {"classifier_name": key}
        if classification.is_nn(key):
            cfg["nn_config"] = self.nn_config()
        elif classification.is_riemann(key):
            cfg["raw_window"] = self.raw_window_value()
            cfg["clf_params"] = self.riemann_params()
        else:
            cfg["clf_params"] = self.classic_params()
        cfg["augment_config"] = self.augment_config()   # viaja con la receta
        return cfg

    def apply_config_dict(self, config: dict) -> bool:
        """Vuelca una configuración en los campos. **No entrena**: solo rellena.

        Devuelve si se aplicó (la config debe ser del clasificador actual)."""
        if not config or config.get("classifier_name") != self.classifier_key:
            return False
        key = self.classifier_key
        if config.get("nn_config"):
            self.nn_config_widget.load_config(config["nn_config"])
            self.update_io_info()
        if config.get("raw_window"):
            self.raw_window.setValue(int(config["raw_window"]))
        # Las configuraciones antiguas no traen aumento: se deja apagado.
        self.set_augment_config(config.get("augment_config"))
        # Tolerante: una config antigua (o de un bundle) puede no traer todas las
        # claves; lo que falte conserva el valor actual del campo.
        p = config.get("clf_params") or {}
        if "multiclass" in p:                    # común a clásicos y Riemann/CSP
            self._select_data(self.multiclass_combo, p["multiclass"])
            self._on_multiclass_changed()
        if key == "random_forest":
            self.rf_estimators.setValue(int(p.get("n_estimators", self.rf_estimators.value())))
            self.rf_max_depth.setValue(int(p.get("max_depth", self.rf_max_depth.value()) or 0))
            self.rf_min_split.setValue(int(p.get("min_samples_split", self.rf_min_split.value())))
            self.rf_min_leaf.setValue(int(p.get("min_samples_leaf", self.rf_min_leaf.value())))
            if "max_features" in p:
                self._select_data(self.rf_max_features, p["max_features"])
            if "criterion" in p:
                self.rf_criterion.setCurrentText(p["criterion"])
            if "class_weight" in p:
                self._select_data(self.rf_class_weight, p["class_weight"])
        elif key == "svm":
            if "kernel" in p:
                self._select_data(self.svm_kernel, p["kernel"])
            self.svm_C.setValue(float(p.get("C", self.svm_C.value())))
            if "gamma" in p:
                self.svm_gamma.setCurrentText(str(p["gamma"]))
            self.svm_degree.setValue(int(p.get("degree", self.svm_degree.value())))
            self.svm_coef0.setValue(float(p.get("coef0", self.svm_coef0.value())))
            if "class_weight" in p:
                self._select_data(self.svm_class_weight, p["class_weight"])
            self._on_svm_kernel_changed()
        elif key == "lda":
            if "solver" in p:
                self.lda_solver.setCurrentText(p["solver"])
            if "shrinkage" in p:
                self._select_data(self.lda_shrinkage, p["shrinkage"])
            self._on_lda_solver_changed()
        return True

    def refresh_model_configs(self) -> None:
        """Repuebla la lista con las configuraciones **guardadas en el proyecto**
        para el clasificador actual. Los valores por defecto de cada modelo son
        los de siempre: aquí solo aparece lo que se haya guardado."""
        if not hasattr(self, "cfg_combo"):
            return
        key = self.classifier_key
        prev = self.cfg_combo.currentText()
        self.cfg_combo.blockSignals(True)
        self.cfg_combo.clear()
        # Siempre disponible: volver a los valores por defecto del programa.
        default = self.default_config_dict(key)
        if default:
            self.cfg_combo.addItem(DEFAULT_CONFIG_NAME, {**default, "is_default": True})
        proj = getattr(self.controller, "project", None)
        entries = [c for c in proj.model_configs()
                   if c.get("classifier_name") == key] if proj is not None else []
        for e in entries:
            self.cfg_combo.addItem(e.get("name", "—"), e)
        i = self.cfg_combo.findText(prev)
        if i >= 0:
            self.cfg_combo.setCurrentIndex(i)
        self.cfg_combo.blockSignals(False)

    def selected_config(self) -> dict | None:
        return self.cfg_combo.currentData() if self.cfg_combo.count() else None

    def load_selected_config(self) -> None:
        cfg = self.selected_config()
        if not cfg:
            return
        if self.apply_config_dict(cfg):
            self._status(f"Configuración «{cfg.get('name')}» cargada en los campos "
                         "(aún sin entrenar).")

    def save_current_config(self) -> None:
        """Guarda los valores actuales como configuración con nombre. No entrena."""
        if getattr(self.controller, "project", None) is None:
            return
        suggested = f"{self.classifier_key}_1"
        name, ok = QInputDialog.getText(self, "Guardar configuración",
                                        "Nombre de la configuración:", text=suggested)
        name = (name or "").strip()
        if not ok or not name:
            return
        if name == DEFAULT_CONFIG_NAME:
            self.controller.warn("Nombre reservado",
                                 "Ese nombre lo usa la entrada de valores por defecto. "
                                 "Elige otro.")
            return
        self.controller.project.save_model_config({**self.current_config_dict(),
                                                   "name": name})
        self.controller.request_autosave()
        self.refresh_model_configs()
        i = self.cfg_combo.findText(name)
        if i >= 0:
            self.cfg_combo.setCurrentIndex(i)
        self._status(f"Configuración «{name}» guardada (sin entrenar).")

    def remove_selected_config(self) -> None:
        cfg = self.selected_config()
        if not cfg or getattr(self.controller, "project", None) is None:
            return
        if cfg.get("is_default"):
            self.controller.warn(
                "Valores por defecto",
                "«{}» no es una configuración guardada: son los valores por defecto "
                "del programa y siempre están disponibles.".format(DEFAULT_CONFIG_NAME))
            return
        self.controller.project.remove_model_config(cfg["name"])
        self.controller.request_autosave()
        self.refresh_model_configs()
        self._status(f"Configuración «{cfg['name']}» eliminada.")

    def _status(self, msg: str) -> None:
        try:
            self.controller.statusBar().showMessage(msg, 4000)
        except Exception:  # noqa: BLE001 — el aviso es opcional
            pass

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        """Selecciona en el combo la opción cuyo ``data`` es ``value``."""
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    @property
    def classifier_key(self) -> str:
        return self.clf_combo.currentData()

    def nn_config(self) -> dict:
        return self.nn_config_widget.config()

    def multiclass_strategy(self) -> str:
        """Estrategia multiclase elegida (nativa / ovo / ovr)."""
        return self.multiclass_combo.currentData() or "nativa"

    def svm_params(self) -> dict:
        return {
            "kernel": self.svm_kernel.currentData(),
            "C": float(self.svm_C.value()),
            "gamma": self.svm_gamma.currentText(),
            "degree": self.svm_degree.value(),
            "coef0": float(self.svm_coef0.value()),
            "class_weight": self.svm_class_weight.currentData(),
            "multiclass": self.multiclass_strategy(),
        }

    def rf_params(self) -> dict:
        return {
            "n_estimators": self.rf_estimators.value(),
            "max_depth": self.rf_max_depth.value(),     # 0 = sin límite
            "min_samples_split": self.rf_min_split.value(),
            "min_samples_leaf": self.rf_min_leaf.value(),
            "max_features": self.rf_max_features.currentData(),
            "criterion": self.rf_criterion.currentText(),
            "class_weight": self.rf_class_weight.currentData(),
            "multiclass": self.multiclass_strategy(),
        }

    def lda_params(self) -> dict:
        return {
            "solver": self.lda_solver.currentText(),
            "shrinkage": self.lda_shrinkage.currentData(),
            "multiclass": self.multiclass_strategy(),
        }

    def riemann_params(self) -> dict:
        """Parámetros de los modelos Riemann/CSP (solo la estrategia multiclase)."""
        return {"multiclass": self.multiclass_strategy()}

    def classic_params(self) -> dict | None:
        """Parámetros del clasificador clásico seleccionado (SVM, RF o LDA)."""
        key = self.classifier_key
        if key == "svm":
            return self.svm_params()
        if key == "random_forest":
            return self.rf_params()
        if key == "lda":
            return self.lda_params()
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
        self.refresh_model_configs()     # las guardadas viven en el proyecto

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
            item.setToolTip(result.split_report())
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
        self.result_label.setText(
            f"{name} — {label}\n{cv}\nClases: {', '.join(r.classes)}\n{r.split_report()}")

    def _activate_selected(self):
        name = self._selected_name()
        if name:
            self.controller.activate_model(name)

    def _metrics_selected(self):
        name = self._selected_name()
        if name:
            self.controller.show_model_metrics(name)

    def _configure_selected(self):
        name = self._selected_name()
        if name:
            self.controller.configure_model(name)

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
